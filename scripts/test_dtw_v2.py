#!/usr/bin/env python3
"""
Fingerspelling DTW - with LANDMARK LINES like LSC70
"""

import cv2
import os
import pickle as pkl
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from collections import deque, Counter
from fastdtw import fastdtw

# Paths
MODEL_PATH = "ml/sign_language_model/data/reference/fingerspelling_by_letter_v2"
MEDIAPIPE_MODEL = "ml/sign_language_model/mediapipe/models/hand_landmarker.task"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Same connections as LSC70
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm
]


def load_templates():
    """Load templates from folder per letter"""
    templates = {}

    for letter in ALPHABET:
        letter_path = os.path.join(MODEL_PATH, letter)
        if not os.path.exists(letter_path):
            continue

        # Load all pickle files for this letter
        pkl_files = glob.glob(os.path.join(letter_path, "*.pickle"))

        if pkl_files:
            data_list = []
            for pf in pkl_files:
                with open(pf, "rb") as f:
                    data_list.append(pkl.load(f))

            templates[letter] = {"data": data_list, "count": len(data_list)}
            print(f"  {letter}: {len(data_list)} samples")

    print(f"\nLoaded {len(templates)} letters")
    return templates


def create_detector():
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_with_lines(frame, hand_landmarks, color=(255, 0, 0)):
    """Draw hand with smaller connected lines"""
    if not hand_landmarks:
        return

    h, w = frame.shape[:2]

    # Draw connections (lines between points) - THINNER
    for idx1, idx2 in HAND_CONNECTIONS:
        pt1 = hand_landmarks[idx1]
        pt2 = hand_landmarks[idx2]

        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), color, 2)  # Thinner lines

    # Draw points (SMALLER circles)
    for lm in hand_landmarks[:21]:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, color, -1)  # Smaller dots


def compare_with_templates(hand_landmarks_list, templates):
    """Compare with DTW"""
    if not hand_landmarks_list or len(hand_landmarks_list) < 21:
        return "", 0.0

    features = np.array(hand_landmarks_list[:21]).flatten()
    features = features.reshape(1, -1)

    best_match = ""
    best_dist = float("inf")

    for letter, template_data in templates.items():
        data_list = template_data["data"]

        for ref_data in data_list:
            ref = np.array(ref_data).flatten().reshape(1, -1)

            dist, _ = fastdtw(features, ref)

            if dist < best_dist:
                best_dist = dist
                best_match = letter

    confidence = max(0, 1.0 - (best_dist / 15.0))

    return best_match, confidence


def main():
    print("=" * 60)
    print("FINGERSPELLING DTW - with LINES")
    print("=" * 60)

    print("\nLoading templates...")
    templates = load_templates()

    print("\nCreating detector...")
    detector = create_detector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Fingerspelling DTW", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fingerspelling DTW", 1280, 720)

    print("=" * 60)
    print("Ready! Press 'q' to quit.")
    print("=" * 60)

    # Buffers - SMALLER for faster response
    letter_buffer = deque(maxlen=8)  # Reduced from 20
    frame_count = 0
    timestamp = 0
    last_letter = ""
    last_conf = 0.0
    stable_count = 0  # Count frames with same letter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        timestamp += 33

        if frame_count % 2 != 0:
            cv2.imshow("Fingerspelling DTW", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = detector.detect_for_video(mp_image, timestamp)

        current_letter = ""
        current_conf = 0.0

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]

            # Convert
            hand_list = [[lm.x, lm.y, lm.z] for lm in hand[:21]]

            # Draw BLUE lines like LSC70
            draw_hand_with_lines(frame, hand, (255, 0, 0))

            # Compare
            letter, conf = compare_with_templates(hand_list, templates)

            if conf > 0.25:  # Lower threshold
                letter_buffer.append((letter, conf))
        else:
            # No hand detected - clear buffer faster
            if len(letter_buffer) > 0:
                letter_buffer.clear()

        # Smoothing - FASTER response
        if len(letter_buffer) >= 2:  # Reduced from 3
            letters = [l for l, c in letter_buffer if c > 0.25]
            if letters:
                most_common = Counter(letters).most_common(1)[0][0]
                count = sum(1 for l in letters if l == most_common)

                # Lower threshold for faster response (40%)
                if count >= len(letter_buffer) * 0.4:
                    if most_common != last_letter:
                        # Letter changed! Update immediately
                        last_letter = most_common
                        last_conf = np.mean(
                            [c for l, c in letter_buffer if l == most_common]
                        )
                        print(f"Changed to: {last_letter} ({last_conf:.2f})")
                    else:
                        last_conf = np.mean(
                            [c for l, c in letter_buffer if l == most_common]
                        )

        # Draw result - lower threshold for display
        if last_letter and last_conf > 0.25:
            cv2.rectangle(frame, (50, 30), (400, 250), (0, 0, 0), -1)
            cv2.rectangle(frame, (50, 30), (400, 250), (0, 200, 0), 5)
            cv2.putText(
                frame,
                last_letter,
                (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                5.0,
                (0, 255, 0),
                10,
            )

            # Confidence bar
            bar_w = int(300 * last_conf)
            cv2.rectangle(frame, (60, 230), (60 + bar_w, 250), (0, 255, 0), -1)

        cv2.imshow("Fingerspelling DTW", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
