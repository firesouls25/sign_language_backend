#!/usr/bin/env python3
"""
Improved fingerspelling camera test with smoothing
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import joblib
from collections import deque

MODEL_PATH = "ml/sign_language_model/data/reference/fingerspelling"
MEDIAPIPE_MODEL = "ml/sign_language_model/mediapipe/models/hand_landmarker.task"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_model():
    model_file = f"{MODEL_PATH}/model.joblib"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None, None


def create_detector():
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    print("Initializing...")

    # Load model
    model, scaler = load_model()
    if model is None:
        print("ERROR: Model not loaded!")
        return

    print(f"Model: {len(model.classes_)} classes")

    # Create detector
    detector = create_detector()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera!")
        detector.close()
        return

    print("Camera ready. Press 'q' to quit.")
    print("Show letters A-Z")

    # Buffers for smoothing
    letter_buffer = deque(maxlen=15)  # More samples for stability
    position_buffer = deque(maxlen=5)

    frame_count = 0
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        timestamp += 33  # ~30fps

        # Process every 3rd frame
        if frame_count % 3 != 0:
            cv2.imshow("Fingerspelling", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Convert
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect
        results = detector.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]

            # Extract features
            features = []
            for lm in hand[:21]:
                features.extend([lm.x, lm.y])
            features = np.array(features).reshape(1, -1)

            # Scale and predict
            scaled = scaler.transform(features)
            proba = model.predict_proba(scaled)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])

            # Only accept if confident enough
            if conf > 0.25:
                letter = ALPHABET[idx] if idx < 26 else "?"
                letter_buffer.append((letter, conf))

                # Track hand position for stability
                center_x = np.mean([lm.x for lm in hand[:21]])
                center_y = np.mean([lm.y for lm in hand[:21]])
                position_buffer.append((center_x, center_y))

        # Draw result if we have enough samples
        if len(letter_buffer) >= 5:
            # Most common letter in buffer
            letters = [l for l, c in letter_buffer if c > 0.2]
            if letters:
                from collections import Counter

                most_common = Counter(letters).most_common(1)[0][0]

                # Get average confidence
                avg_conf = np.mean([c for l, c in letter_buffer if l == most_common])

                # Draw on frame
                cv2.putText(
                    frame,
                    f"{most_common} ({avg_conf:.2f})",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

        # Draw hand landmarks
        if results.hand_landmarks:
            h, w = frame.shape[:2]
            for lm in results.hand_landmarks[0][:21]:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Fingerspelling", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done!")


if __name__ == "__main__":
    main()
