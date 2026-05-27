import os
import logging
import numpy as np
from typing import Dict
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_MAP = {i: letter for i, letter in enumerate(ALPHABET)}


class FingerspellingRecognizer:
    """
    Recognizer for ASL fingerspelling.
    Uses a small 2-layer MLP (63→128→26) with numpy-only inference.

    Model architecture (trained in Keras, exported to .npz weights):
      Input: 21 landmarks x 3 coords (x, y, z) = 63 features
      Dense(128, relu) → Dense(26, softmax)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_loaded = False
        self.load_error = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.letter_buffer = deque(maxlen=10)
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.cooldown = timedelta(seconds=0.3)
        self.confidence_threshold = 0.3
        self.letter_history = []

        self._load_model()

    def _load_model(self):
        """Load pre-exported .npz weights (no TensorFlow needed)."""
        npz_file = os.path.join(self.model_path, "asl-now-weights.npz")
        logger.info(f"[FingerspellingRecognizer] Looking for weights: {npz_file}")

        if not os.path.exists(npz_file):
            self.load_error = f"Weights file not found: {npz_file}"
            logger.error(f"[FingerspellingRecognizer] {self.load_error}")
            self.is_loaded = False
            return

        try:
            data = np.load(npz_file)
            self.W1 = data["layer_1_0"]
            self.b1 = data["layer_1_1"]
            self.W2 = data["layer_2_0"]
            self.b2 = data["layer_2_1"]
            data.close()
            self.is_loaded = True
            logger.info(
                f"[FingerspellingRecognizer] Weights loaded: W1={self.W1.shape}, "
                f"b1={self.b1.shape}, W2={self.W2.shape}, b2={self.b2.shape}"
            )
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"[FingerspellingRecognizer] Failed to load weights: {e}")

    def _forward(self, features: np.ndarray) -> tuple:
        """
        Forward pass through the MLP.
        features: (1, 63) flattened landmarks
        Returns: (letter_idx, confidence)
        """
        h = features @ self.W1 + self.b1
        h = np.maximum(h, 0)
        logits = h @ self.W2 + self.b2
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        probs = probs[0]
        letter_idx = int(np.argmax(probs))
        confidence = float(probs[letter_idx])
        return letter_idx, confidence

    def extract_features(self, landmarks) -> np.ndarray:
        """Extract flattened (1, 63) feature vector from 21 hand landmarks."""
        if not landmarks or len(landmarks) < 21:
            return None

        flat = []
        for i in range(21):
            point = landmarks[i] if i < len(landmarks) else [0.0, 0.0, 0.0]
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            z = float(point[2]) if len(point) > 2 else 0.0
            flat.extend([x, y, z])

        return np.array(flat, dtype=np.float32).reshape(1, -1)

    def predict(self, left_landmarks, right_landmarks) -> Dict:
        logger.info(f"[FingerspellingRecognizer] predict() called")
        logger.info(f"[FingerspellingRecognizer] is_loaded: {self.is_loaded}")
        logger.info(f"[FingerspellingRecognizer] left_landmarks: {left_landmarks is not None}")
        logger.info(f"[FingerspellingRecognizer] right_landmarks: {right_landmarks is not None}")

        if not self.is_loaded:
            logger.info(f"[FingerspellingRecognizer] Using placeholder (model not loaded)")
            return self._placeholder_predict(left_landmarks, right_landmarks)

        try:
            landmarks = (
                right_landmarks
                if right_landmarks and len(right_landmarks) >= 21
                else left_landmarks
            )

            if not landmarks or len(landmarks) < 21:
                logger.info(f"[FingerspellingRecognizer] No valid landmarks")
                return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": "", "sequence": ""}

            logger.info(f"[FingerspellingRecognizer] Extracting features for {len(landmarks)} points")
            features = self.extract_features(landmarks)

            if features is None:
                logger.info(f"[FingerspellingRecognizer] Feature extraction returned None")
                return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": "", "sequence": ""}

            logger.info(f"[FingerspellingRecognizer] Running model prediction...")
            letter_idx, confidence = self._forward(features)
            letter = LABEL_MAP.get(letter_idx, "?")

            logger.info(f"[FingerspellingRecognizer] Prediction: {letter} (idx={letter_idx}) confidence={confidence:.4f}")

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
                    logger.warning(f"[FingerspellingRecognizer] Best letter from buffer: {best_letter} (count={letter_counts[best_letter]})")

                    now = datetime.now()
                    time_diff = (now - self.last_detection_time).total_seconds()
                    logger.warning(f"[Fingerspelling] time_diff: {time_diff}s, cooldown: {self.cooldown.total_seconds()}s")

                    if best_letter != self.last_letter or time_diff > self.cooldown.total_seconds():
                        self.last_letter = best_letter
                        self.last_detection_time = now
                        self.letter_history.append(best_letter)
                        logger.warning(f"[Fingerspelling] LETTER ADDED: {best_letter}, history: {self.letter_history}")

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
            return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": "", "sequence": ""}

    def _placeholder_predict(self, left_landmarks, right_landmarks) -> Dict:
        logger.info(f"[FingerspellingRecognizer] _placeholder_predict() called")
        has_left = left_landmarks and len(left_landmarks) >= 21
        has_right = right_landmarks and len(right_landmarks) >= 21
        has_any = has_left or has_right

        logger.info(f"[FingerspellingRecognizer] has_landmarks: {has_any}")
        logger.info(f"[FingerspellingRecognizer] load_error: {self.load_error}")

        if not has_any:
            return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": "", "sequence": ""}

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
        self.letter_buffer.clear()
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.letter_history = []
