#!/usr/bin/env python3
"""
Script de comparación: cámara → predicción Keras directa vs cámara → WebSocket backend.
Útil para verificar si el modelo se behaves the same as when the local test.

Usage:
    python scripts/test_camera_comparison.py
"""

import cv2
import json
import logging
import mediapipe as mp
import numpy as np
import os
import sys
import time
import threading
import websocket
from datetime import datetime
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "sign_language_model")
sys.path.insert(0, ML_DIR)
os.chdir(ML_DIR)

MODEL_PATH = "data/reference/fingerspelling/asl-now-weights.h5"
HAND_MODEL = "mediapipe/models/hand_landmarker.task"
WS_URL = "ws://localhost:8000/ws/translate"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    import tensorflow as tf

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


def create_hand_detector(model_path):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
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


def landmarks_to_list(landmarks):
    if not landmarks:
        return []
    return [[lm.x, lm.y, lm.z] for lm in landmarks]


def predict_direct(model, landmarks, letter_buffer, last_letter, last_time):
    if not landmarks or len(landmarks) < 21:
        return "", "", False, 0.0

    features = np.array(
        [[[lm.x, lm.y, lm.z] for lm in landmarks[:21]]], dtype=np.float32
    )
    pred = model.predict(features, verbose=0)[0]

    letter_idx = int(np.argmax(pred))
    confidence = float(pred[letter_idx])
    letter = ALPHABET[letter_idx]

    if confidence > 0.1:
        letter_buffer.append((letter, confidence))

        if len(letter_buffer) >= 3:
            counts = {}
            for l, c in letter_buffer:
                counts[l] = counts.get(l, 0) + 1

            best_letter = max(counts, key=counts.get)
            now = datetime.now()

            if best_letter != last_letter or (now - last_time).total_seconds() > 0.5:
                return best_letter, best_letter, True, confidence

    return "", letter, True, confidence


class WebSocketClient:
    def __init__(self, url=WS_URL):
        self.url = url
        self.ws = None
        self.connected = False
        self.last_backend_response = None
        self._response_ready = threading.Event()

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self._on_message,
            on_open=self._on_open,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

        for _ in range(20):
            if self.connected:
                return True
            time.sleep(0.1)

        return False

    def _on_open(self, ws):
        self.connected = True
        logger.info("[WS] Connected")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.last_backend_response = data
            self._response_ready.set()
        except:
            pass

    def _on_error(self, ws, error):
        logger.error(f"[WS] Error: {error}")

    def _on_close(self, ws, *args):
        self.connected = False
        logger.info("[WS] Disconnected")

    def send_mode(self, mode):
        if not self.ws:
            return
        message = {"type": "set_mode", "mode": mode}
        self.ws.send(json.dumps(message))

    def send_landmarks(self, landmarks):
        if not self.ws or not landmarks:
            return

        message = {
            "type": "landmarks",
            "data": {"left_hand": landmarks, "right_hand": [], "pose": []},
            "mode": "fingerspelling",
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

        self.ws.send(json.dumps(message))
        self._response_ready.clear()

    def send_reset(self):
        if not self.ws:
            return
        message = {"type": "reset"}
        self.ws.send(json.dumps(message))

    def get_response(self, timeout=2.0):
        self._response_ready.wait(timeout)
        return self.last_backend_response

    def close(self):
        if self.ws:
            self.ws.close()


def draw_comparison(frame, direct_result, backend_result, has_hand):
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        "COMPARISON: Keras Local vs Backend",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 200, 255),
        2,
    )

    direct_letter = direct_result[0] or direct_result[1] or ""
    direct_conf = direct_result[3]

    cv2.putText(
        frame,
        f"Direct: {direct_letter or '?'} ({direct_conf:.0%})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if backend_result:
        be_text = backend_result.get("candidate", "")
        be_conf = backend_result.get("confidence", 0.0)
        cv2.putText(
            frame,
            f"Backend: {be_text or '?'} ({be_conf:.0%})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 100, 100),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Backend: (waiting...)",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (150, 150, 150),
            2,
        )

    color = (0, 255, 0) if has_hand else (100, 100, 100)
    cv2.circle(frame, (w - 40, 40), 15, color, -1)

    cv2.putText(
        frame,
        "q=quit",
        (w - 120, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )

    return frame


def main():
    print("=" * 60)
    print("COMPARISON: Camera Keras Local vs WebSocket Backend")
    print("=" * 60)

    print("\n[1] Loading Keras model...")
    model = load_model()
    print("    Model loaded!")

    print("\n[2] Loading MediaPipe detector...")
    detector = create_hand_detector(HAND_MODEL)
    print("    Detector ready!")

    print("\n[3] Connecting to WebSocket backend...")
    ws_client = WebSocketClient(WS_URL)
    if not ws_client.connect():
        print("    ERROR: Cannot connect to backend")
        return
    print("    Connected!")

    ws_client.send_mode("fingerspelling")
    ws_client.send_reset()

    print("\n[4] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("    ERROR: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("    Camera ready!")

    print("\n" + "=" * 60)
    print("COMPARING: Press 'q' to quit")
    print("=" * 60 + "\n")

    frame_count = 0
    letter_buffer = deque(maxlen=5)
    last_letter = ""
    last_time = datetime.min

    backend_response = None
    direct_result = ("", "", False, 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        landmarks = None
        has_hand = False

        if frame_count % 2 == 0:
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            landmarks, hand_side = detect_hand(frame, detector, timestamp_ms)
            has_hand = landmarks is not None

            if landmarks:
                landmarks_list = landmarks_to_list(landmarks)

                direct_result = predict_direct(
                    model, landmarks, letter_buffer, last_letter, last_time
                )

                if direct_result[0]:
                    last_letter = direct_result[0]
                    last_time = datetime.now()

                ws_client.send_landmarks(landmarks_list)
                backend_response = ws_client.get_response(timeout=0.5)

        draw_result = direct_result if has_hand else ("", "", False, 0.0)
        frame = draw_comparison(frame, draw_result, backend_response, has_hand)
        cv2.imshow("Comparison Local vs Backend", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    ws_client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
