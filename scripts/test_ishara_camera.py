#!/usr/bin/env python3
"""
Test fingerspelling model with webcam.
Uses MediaPipe for hand landmarks and Keras model for prediction.
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from collections import deque

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "sign_language_model")
sys.path.insert(0, ML_DIR)
os.chdir(ML_DIR)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections, drawing_utils

MODEL_PATH = "data/reference/fingerspelling/asl-now-weights.h5"
HAND_MODEL = "mediapipe/models/hand_landmarker.task"

WINDOW_NAME = "Fingerspelling Recognition"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(21, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(26, activation="softmax"),
        ]
    )
    model.load_weights(MODEL_PATH)
    return model


def create_hand_detector(model_path, running_mode=vision.RunningMode.VIDEO):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=running_mode,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand(image, detector, timestamp_ms):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results and results.hand_landmarks:
        for idx in range(len(results.hand_landmarks)):
            handedness = results.handedness[idx][0].category_name
            if handedness == "Left":
                return results.hand_landmarks[idx], "left"
            elif handedness == "Right":
                return results.hand_landmarks[idx], "right"
        return results.hand_landmarks[0], "right"

    return None, None


def draw_hand(frame, landmarks):
    if not landmarks:
        return frame

    h, w = frame.shape[:2]
    drawing_utils.draw_landmarks(
        frame,
        landmark_list=landmarks,
        connections=HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_utils.DrawingSpec(
            color=(0, 255, 255), thickness=3, circle_radius=4
        ),
        connection_drawing_spec=drawing_utils.DrawingSpec(
            color=(0, 200, 0), thickness=2, circle_radius=2
        ),
    )
    return frame


def predict_letter(model, landmarks, letter_buffer, last_letter, last_time, cooldown):
    if not landmarks or len(landmarks) < 21:
        return "", "", False

    features = np.array(
        [[[lm.x, lm.y, lm.z] for lm in landmarks[:21]]], dtype=np.float32
    )
    pred = model.predict(features, verbose=0)[0]

    letter_idx = int(np.argmax(pred))
    confidence = float(pred[letter_idx])
    letter = ALPHABET[letter_idx]

    if confidence > 0.3:
        letter_buffer.append((letter, confidence))

        if len(letter_buffer) >= 3:
            counts = {}
            for l, c in letter_buffer:
                counts[l] = counts.get(l, 0) + 1

            best = max(counts, key=counts.get)
            now = datetime.now()

            if best != last_letter or (now - last_time).total_seconds() > cooldown:
                return best, best, True

    return "", letter, True


def draw_ui(frame, current_word="", candidate="", has_hand=False, confidence=0.0):
    h, w = frame.shape[:2]

    bottom_h = 110
    bottom_y = h - bottom_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bottom_y), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (0, bottom_y), (w, h), (70, 70, 70), 2)

    cv2.putText(
        frame,
        "Modelo: Keras ASL",
        (w - 220, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 180, 100),
        2,
    )

    if current_word:
        cv2.putText(
            frame,
            f"PALABRA: {current_word}",
            (30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (50, 220, 80),
            3,
        )
    elif candidate and has_hand:
        cv2.putText(
            frame,
            f"Letra: {candidate}? ({confidence:.0%})",
            (30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (220, 200, 80),
            3,
        )
    elif has_hand:
        cv2.putText(
            frame,
            "Detectando mano...",
            (30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (150, 150, 150),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Muestra tu mano",
            (30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (180, 180, 180),
            2,
        )

    color = (0, 255, 0) if has_hand else (100, 100, 100)
    cv2.circle(frame, (w - 50, 70), 18, color, -1)

    cv2.putText(
        frame,
        "q=salir  r=reset",
        (w - 200, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (130, 130, 130),
        1,
    )
    return frame


def main():
    print("=" * 50)
    print("Fingerspelling Test")
    print("=" * 50)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Loading model...")
    model = load_model()
    print("  Model loaded!")

    print("Loading hand detector...")
    detector = create_hand_detector(HAND_MODEL)
    print("  Detector ready!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nListo! Presiona 'q' para salir, 'r' para reiniciar.\n")

    frame_count = 0
    current_word = ""
    last_letter = ""
    last_time = datetime.min
    cooldown = 0.5
    letter_buffer = deque(maxlen=5)
    last_candidate = ""
    last_confidence = 0.0
    has_hand = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        landmarks = None
        hand_side = None

        if frame_count % 2 == 0:
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            landmarks, hand_side = detect_hand(frame, detector, timestamp_ms)
            has_hand = landmarks is not None

            frame = draw_hand(frame, landmarks)

            if landmarks and model:
                word, candidate, recording = predict_letter(
                    model, landmarks, letter_buffer, last_letter, last_time, cooldown
                )

                if word:
                    current_word += word
                    last_letter = word
                    last_time = datetime.now()
                    print(f"Letra: {word}")

                feat = np.array(
                    [[[lm.x, lm.y, lm.z] for lm in landmarks[:21]]], dtype=np.float32
                )
                proba = model.predict(feat, verbose=0)[0]
                letter_idx = int(np.argmax(proba))
                last_candidate = ALPHABET[letter_idx]
                last_confidence = float(proba[letter_idx])

        frame = draw_ui(frame, current_word, last_candidate, has_hand, last_confidence)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            current_word = ""
            last_letter = ""
            letter_buffer.clear()
            last_candidate = ""
            last_confidence = 0.0
            print("Reiniciado!")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Listo!")


if __name__ == "__main__":
    main()
