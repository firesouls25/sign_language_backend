import os
import logging
import numpy as np
from typing import Optional, Dict, List
import joblib

logger = logging.getLogger(__name__)


class HandshapeRecognizer:
    """
    Recognizer for static handshapes (ASL signs).
    Uses an MLP classifier trained on ASL Signs dataset.
    Maps labels to Spanish concepts.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """Load trained model, scaler and label encoder"""
        model_file = os.path.join(self.model_path, "model.joblib")
        labels_file = os.path.join(self.model_path, "labels.npy")

        if os.path.exists(model_file) and os.path.exists(labels_file):
            try:
                self.model, self.scaler = joblib.load(model_file)
                self.label_encoder = np.load(labels_file, allow_pickle=True).item()
                self.is_loaded = True
                logger.info(
                    f"[HandshapeRecognizer] Loaded model with {len(self.label_encoder)} classes"
                )
                logger.info(
                    f"[HandshapeRecognizer] Classes: {list(self.label_encoder.keys())[:10]}..."
                )
            except Exception as e:
                logger.error(f"[HandshapeRecognizer] Error loading model: {e}")
                self.is_loaded = False
        else:
            logger.warning(f"[HandshapeRecognizer] Model files not found at {self.model_path}")
            self.is_loaded = False

    def extract_features(self, left_landmarks, right_landmarks) -> np.ndarray:
        """Extract features from hand landmarks (21 points x 3 coords = 63 per hand)"""
        left_flat = []
        right_flat = []

        if left_landmarks and len(left_landmarks) >= 21:
            for point in left_landmarks[:21]:
                left_flat.extend([point[0], point[1], point[2] if len(point) > 2 else 0.0])
        else:
            left_flat = [0.0] * 63

        if right_landmarks and len(right_landmarks) >= 21:
            for point in right_landmarks[:21]:
                right_flat.extend([point[0], point[1], point[2] if len(point) > 2 else 0.0])
        else:
            right_flat = [0.0] * 63

        # Use right hand primarily (most people fingerspell with dominant hand)
        features = right_flat if right_landmarks and len(right_landmarks) >= 21 else left_flat
        return np.array(features).reshape(1, -1)

    def predict(self, left_landmarks, right_landmarks) -> Dict:
        """Predict handshape and return concept"""
        if not self.is_loaded or self.model is None:
            return self._placeholder_predict(left_landmarks, right_landmarks)

        try:
            features = self.extract_features(left_landmarks, right_landmarks)

            has_hands = (left_landmarks is not None and len(left_landmarks) >= 21) or (
                right_landmarks is not None and len(right_landmarks) >= 21
            )

            if not has_hands:
                return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": ""}

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get prediction
            prediction_idx = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))

            # Get label from encoder (idx -> label)
            idx_to_label = {v: k for k, v in self.label_encoder.items()}
            label = idx_to_label.get(prediction_idx, str(prediction_idx))

            return {
                "text": label.upper(),
                "confidence": confidence,
                "is_recording": False,
                "candidate": label.upper(),
            }

        except Exception as e:
            logger.error(f"[HandshapeRecognizer] Prediction error: {e}")
            return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": ""}

    def _placeholder_predict(self, left_landmarks, right_landmarks) -> Dict:
        """Placeholder prediction when model is not trained yet"""
        has_left = left_landmarks and len(left_landmarks) >= 21
        has_right = right_landmarks and len(right_landmarks) >= 21

        if not has_left and not has_right:
            return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": ""}

        return {"text": "A", "confidence": 0.5, "is_recording": False, "candidate": "A"}

    def reset(self):
        """Reset recognizer state"""
        pass
