import os
import logging
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_MAP = {i: letter for i, letter in enumerate(ALPHABET)}

# Import sklearn for custom model
try:
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(f"[FingerspellingRecognizer] sklearn/joblib not available")


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
        self.load_error = None

        self.letter_buffer = deque(maxlen=10)
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.cooldown = timedelta(seconds=0.3)
        self.confidence_threshold = 0.3  # Same as test scripts for consistency
        self.letter_history = []

        self._load_model()

    def _load_model(self):
        """Load Keras model - imports TensorFlow lazily to save memory."""
        weights_file = os.path.join(self.model_path, "asl-now-weights.h5")
        keras_file = os.path.join(self.model_path, "asl-now.keras")

        logger.info(f"[FingerspellingRecognizer] Attempting to load TensorFlow/Keras model")

        # Lazy import TensorFlow - saves ~200-400MB RAM when TF isn't needed
        try:
            import tensorflow as tf
        except ImportError:
            self.load_error = "TensorFlow not available"
            logger.error("[FingerspellingRecognizer] Cannot load model: TensorFlow not available")
            self.is_loaded = False
            return

        try:
            if os.path.exists(weights_file):
                logger.info(f"[FingerspellingRecognizer] Loading Keras weights from {weights_file}")
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
                self.model_type = "tensorflow"
                logger.info("[FingerspellingRecognizer] SUCCESS: TensorFlow Keras loaded!")
                return
            elif os.path.exists(keras_file):
                logger.info(f"[FingerspellingRecognizer] Loading .keras model from {keras_file}")
                self.model = tf.keras.models.load_model(keras_file)
                self.is_loaded = True
                self.model_type = "tensorflow"
                logger.info("[FingerspellingRecognizer] SUCCESS: .keras model loaded!")
                return
            else:
                self.load_error = f"No model file found at {self.model_path}"
                logger.error(f"[FingerspellingRecognizer] No model files found")
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"[FingerspellingRecognizer] Failed to load Keras: {e}")

        logger.warning(f"[FingerspellingRecognizer] Model loading failed")
        self.is_loaded = False

    def extract_features(self, landmarks) -> np.ndarray:
        """Extract features from hand landmarks."""
        if not landmarks or len(landmarks) < 21:
            return None

        # Extract x,y for each point (in sequence: x1,y1,x2,y2,...x21,y21 = 42 values)
        features = []
        for i in range(21):
            point = landmarks[i] if i < len(landmarks) else [0.0, 0.0, 0.0]
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            features.extend([x, y])

        features = np.array(features).reshape(1, -1)

        # tensorflow model uses 63 features (x,y,z for 21 points)
        if getattr(self, "model_type", None) != "sklearn":
            # For tensorflow, we need x,y,z format
            tf_features = []
            for i in range(21):
                point = landmarks[i] if i < len(landmarks) else [0.0, 0.0, 0.0]
                x = float(point[0]) if len(point) > 0 else 0.0
                y = float(point[1]) if len(point) > 1 else 0.0
                z = float(point[2]) if len(point) > 2 else 0.0
                tf_features.append([x, y, z])
            return np.array(tf_features).reshape(1, 21, 3)

        return features

    def predict(self, left_landmarks, right_landmarks) -> Dict:
        """Predict letter from hand landmarks."""
        logger.info(f"[FingerspellingRecognizer] predict() called")
        logger.info(f"[FingerspellingRecognizer] is_loaded: {self.is_loaded}")
        logger.info(f"[FingerspellingRecognizer] left_landmarks: {left_landmarks is not None}")
        logger.info(f"[FingerspellingRecognizer] right_landmarks: {right_landmarks is not None}")

        if not self.is_loaded or self.model is None:
            logger.info(f"[FingerspellingRecognizer] Using placeholder (model not loaded)")
            return self._placeholder_predict(left_landmarks, right_landmarks)

        try:
            # Prefer right hand for fingerspelling (dominant hand)
            landmarks = (
                right_landmarks
                if right_landmarks and len(right_landmarks) >= 21
                else left_landmarks
            )

            if not landmarks or len(landmarks) < 21:
                logger.info(f"[FingerspellingRecognizer] No valid landmarks")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "is_recording": False,
                    "candidate": "",
                    "sequence": "",
                }

            logger.info(
                f"[FingerspellingRecognizer] Extracting features for {len(landmarks)} points"
            )
            features = self.extract_features(landmarks)

            if features is None:
                logger.info(f"[FingerspellingRecognizer] Feature extraction returned None")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "is_recording": False,
                    "candidate": "",
                    "sequence": "",
                }

            logger.info(f"[FingerspellingRecognizer] Running model prediction...")

            if getattr(self, "model_type", None) == "sklearn" and hasattr(self, "scaler"):
                # sklearn prediction
                features_scaled = self.scaler.transform(features)
                proba = self.model.predict_proba(features_scaled)[0]
                letter_idx = int(np.argmax(proba))
                confidence = float(proba[letter_idx])
            else:
                # TensorFlow prediction
                predictions = self.model.predict(features, verbose=0)[0]
                letter_idx = int(np.argmax(predictions))
                confidence = float(predictions[letter_idx])

            letter = LABEL_MAP.get(letter_idx, "?")

            logger.info(
                f"[FingerspellingRecognizer] Prediction: {letter} (idx={letter_idx}) confidence={confidence:.4f}"
            )

            self.letter_buffer.append((letter, confidence))

            if len(self.letter_buffer) >= 3:
                letter_counts = {}
                for l, c in self.letter_buffer:
                    if c >= self.confidence_threshold:
                        letter_counts[l] = letter_counts.get(l, 0) + 1

                logger.warning(f"[Fingerspelling] letter_buffer: {list(self.letter_buffer)}")
                logger.warning(f"[Fingerspelling] letter_counts: {letter_counts}")

                if letter_counts:
                    best_letter = max(letter_counts, key=letter_counts.get)
                    logger.warning(
                        f"[FingerspellingRecognizer] Best letter from buffer: {best_letter} (count={letter_counts[best_letter]})"
                    )

                    now = datetime.now()
                    time_diff = (now - self.last_detection_time).total_seconds()
                    logger.warning(
                        f"[Fingerspelling] time_diff: {time_diff}s, cooldown: {self.cooldown.total_seconds()}s"
                    )

                    if best_letter != self.last_letter or time_diff > self.cooldown.total_seconds():
                        self.last_letter = best_letter
                        self.last_detection_time = now
                        self.letter_history.append(best_letter)
                        logger.warning(
                            f"[Fingerspelling] LETTER ADDED: {best_letter}, history: {self.letter_history}"
                        )

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
            import traceback

            logger.error(f"[FingerspellingRecognizer] Traceback: {traceback.format_exc()}")
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
                "sequence": "",
            }

    def _placeholder_predict(self, left_landmarks, right_landmarks) -> Dict:
        """Placeholder when model not loaded - returns confidence 0.0 to distinguish from real predictions"""
        logger.info(f"[FingerspellingRecognizer] _placeholder_predict() called")
        has_left = left_landmarks and len(left_landmarks) >= 21
        has_right = right_landmarks and len(right_landmarks) >= 21
        has_any = has_left or has_right

        logger.info(f"[FingerspellingRecognizer] has_landmarks: {has_any}")
        logger.info(f"[FingerspellingRecognizer] load_error: {self.load_error}")

        if not has_any:
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
                "sequence": "",
            }

        # Return 0.0 confidence to indicate this is from placeholder
        # This helps distinguish from real predictions
        return {
            "text": "",
            "confidence": 0.0,
            "is_recording": True,
            "candidate": "A",
            "sequence": "",
            "IS_PLACEHOLDER": True,
            "load_error": self.load_error,
        }

    def reset(self):
        """Reset recognizer state"""
        self.letter_buffer.clear()
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.letter_history = []
