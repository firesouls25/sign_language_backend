#!/usr/bin/env python3
"""
Test fingerspelling model with webcam in real-time
Uses MediaPipe Tasks API
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import joblib

MODEL_PATH = "ml/sign_language_model/data/reference/fingerspelling"
MEDIAPIPE_MODEL = "ml/sign_language_model/mediapipe/models/hand_landmarker.task"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_model():
    """Load sklearn model"""
    model_file = f"{MODEL_PATH}/model.joblib"
    if os.path.exists(model_file):
        model, scaler = joblib.load(model_file)
        return model, scaler
    return None, None


def create_detector():
    """Create MediaPipe hand detector"""
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_features(hand_landmarks):
    """Extract x,y features from hand landmarks"""
    features = []
    for lm in hand_landmarks[:21]:
        features.extend([lm.x, lm.y])
    return np.array(features).reshape(1, -1)


def predict(model, scaler, hand_landmarks):
    """Predict letter"""
    if model is None or scaler is None or not hand_landmarks:
        return "", 0.0

    features = extract_features(hand_landmarks)
    scaled = scaler.transform(features)
    proba = model.predict_proba(scaled)[0]
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])

    letter = ALPHABET[idx] if idx < len(ALPHABET) else "?"
    return letter, confidence


def main():
    print("Initializing...")

    # Load model
    model, scaler = load_model()
    if model is None:
        print("ERROR: Model not loaded!")
        return

    print(f"Model loaded: {len(model.classes_)} classes")

    # Create detector
    detector = create_detector()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        detector.close()
        return

    print("Camera opened. Press 'q' to quit.")
    print("Show hand letters A-Z to test!")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Process every 3rd frame for speed
        if frame_count % 3 == 0:
            # Convert to MediaPipe image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect hands
            results = detector.detect_for_video(
                mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
            )

            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand_landmarks = results.hand_landmarks[0]

                # Draw landmarks (simple circles)
                h, w = frame.shape[:2]
                for lm in hand_landmarks[:21]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Predict
                letter, conf = predict(model, scaler, hand_landmarks)

                if letter and conf > 0.15:
                    cv2.putText(
                        frame,
                        f"{letter} ({conf:.2f})",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )

        cv2.imshow("Fingerspelling Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done!")


if __name__ == "__main__":
    main()
