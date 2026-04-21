#!/usr/bin/env python3
"""
Fingerspelling DTW - improved version with better stability
Similar to LSC70 system
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "ml/sign_language_model/data/reference/fingerspelling_pickles"
MEDIAPIPE_MODEL = "ml/sign_language_model/mediapipe/models/hand_landmarker.task"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_templates():
    """Load all reference templates from pickle files"""
    templates = {}
    lh_files = glob.glob(f"{MODEL_PATH}/lh_*.pickle")

    logger.info(f"Loading templates from {MODEL_PATH}")

    for lh_file in lh_files:
        basename = os.path.basename(lh_file)
        letter = basename.split("_")[1]

        with open(lh_file, "rb") as f:
            left_data = pkl.load(f)

        rh_file = lh_file.replace("lh_", "rh_")
        if os.path.exists(rh_file):
            with open(rh_file, "rb") as f:
                right_data = pkl.load(f)
        else:
            right_data = [[0.0] * 63] * len(left_data)

        templates[letter] = {
            "left": left_data,
            "right": right_data,
            "count": len(left_data),
        }

    logger.info(f"Loaded {len(templates)} letters: {list(templates.keys())}")

    # Show counts
    for letter, data in templates.items():
        logger.info(f"  {letter}: {data['count']} samples")

    return templates


def create_detector():
    """Create MediaPipe hand detector"""
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def compare_with_templates(hand_landmarks_list, templates):
    """Compare with DTW - returns best match and confidence"""
    if not hand_landmarks_list or len(hand_landmarks_list) < 21:
        return "", 0.0

    # Convert to numpy array
    features = np.array(hand_landmarks_list[:21]).flatten()
    features = features.reshape(1, -1)

    best_match = ""
    best_dist = float("inf")
    all_distances = []

    # Compare with ALL templates
    for letter, template_data in templates.items():
        left_templates = template_data["left"]
        right_templates = template_data["right"]

        # Try all samples for this letter
        for i in range(len(left_templates)):
            left_ref = np.array(left_templates[i]).flatten().reshape(1, -1)
            right_ref = np.array(right_templates[i]).flatten().reshape(1, -1)

            # DTW distances
            dist_l, _ = fastdtw(features, left_ref)
            dist_r, _ = fastdtw(features, right_ref)

            dist = min(dist_l, dist_r)
            all_distances.append((letter, dist))

            if dist < best_dist:
                best_dist = dist
                best_match = letter

    # Confidence based on distance
    # Normalize: closer = higher confidence
    confidence = max(0, 1.0 - (best_dist / 15.0))

    # Log top matches occasionally
    if len(all_distances) > 0:
        sorted_dists = sorted(all_distances, key=lambda x: x[1])[:3]
        top_str = ", ".join([f"{l}({d:.2f})" for l, d in sorted_dists])
        logger.debug(f"Top: {top_str}")

    return best_match, confidence


def main():
    print("=" * 60)
    print("FINGERSPELLING DTW - LSC70 Style")
    print("=" * 60)

    # Load templates
    print("\n[1/3] Loading templates...")
    templates = load_templates()
    print(f"    ✓ {len(templates)} letters loaded")

    # Create detector
    print("\n[2/3] Creating MediaPipe detector...")
    detector = create_detector()
    print("    ✓ Detector ready")

    # Open camera
    print("\n[3/3] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Can't open camera!")
        detector.close()
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Windows
    cv2.namedWindow("Fingerspelling DTW", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fingerspelling DTW", 1280, 720)

    print("    ✓ Camera ready (1280x720)")
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("- Show hand letters A-Z to camera")
    print("- Hold steady for best recognition")
    print("- Press 'q' to quit")
    print("=" * 60)

    # Buffers for stability - LARGER to reduce flickering
    letter_buffer = deque(maxlen=30)  # Much larger for stability
    dist_buffer = deque(maxlen=30)
    frame_count = 0
    timestamp = 0

    last_show_letter = ""
    last_show_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        frame_count += 1
        timestamp += 33

        # Process every 3rd frame for stability
        if frame_count % 3 != 0:
            cv2.imshow("Fingerspelling DTW", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Convert frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hands
        results = detector.detect_for_video(mp_image, timestamp)

        current_letter = ""
        current_conf = 0.0

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]

            # Convert landmarks
            hand_list = [[lm.x, lm.y, lm.z] for lm in hand[:21]]

            # Compare with templates
            letter, conf = compare_with_templates(hand_list, templates)

            # Only accept if confidence is good enough
            if conf > 0.35:  # Higher threshold for stability
                letter_buffer.append(letter)
                dist_buffer.append(conf)
                current_letter = letter
                current_conf = conf

                logger.info(f"Detected: {letter} Confidence: {conf:.2f}")

        # Smoothing - require MINIMUM 5 samples of same letter
        if len(letter_buffer) >= 5:
            # Get most common letter
            letters = [l for l in letter_buffer if l]  # Filter empty
            if letters:
                counts = Counter(letters)
                most_common_letter, count = counts.most_common(1)[0]

                # Require at least 60% of buffer to be same letter
                if count >= len(letter_buffer) * 0.6:
                    last_show_letter = most_common_letter
                    last_show_conf = np.mean(
                        [
                            c
                            for l, c in zip(letter_buffer, dist_buffer)
                            if l == most_common_letter
                        ]
                    )
                else:
                    # Not stable enough
                    last_show_letter = "?"
                    last_show_conf = 0.0

        # Draw big text
        if last_show_letter and last_show_letter != "?":
            # Black background box
            cv2.rectangle(frame, (50, 30), (450, 280), (0, 0, 0), -1)
            cv2.rectangle(frame, (50, 30), (450, 280), (0, 255, 0), 5)

            # Letter
            cv2.putText(
                frame,
                last_show_letter,
                (100, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                6.0,
                (0, 255, 0),
                10,
            )

            # Confidence bar
            bar_width = int(350 * last_show_conf)
            cv2.rectangle(frame, (60, 250), (60 + bar_width, 270), (0, 255, 0), -1)

        # Draw hand landmarks
        if results.hand_landmarks:
            h, w = frame.shape[:2]
            for lm in results.hand_landmarks[0][:21]:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)

        # Status text
        cv2.putText(
            frame,
            f"Buffer: {len(letter_buffer)}/30",
            (50, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Fingerspelling DTW", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
