#!/usr/bin/env python3
"""
Extract landmarks to pickles - organized by letter
folder structure: output/{letter}/lh_{sample}.pickle
"""

import cv2
import os
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


# MediaPipe hand connections for drawing
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


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image, detector):
    """Extract hand landmarks with proper format"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = detector.detect(mp_image)

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if results and results.hand_landmarks:
        for idx, hand_classification in enumerate(results.handedness):
            if len(hand_classification) > 0:
                hand_label = hand_classification[0].category_name
                landmarks = results.hand_landmarks[idx]
                points = [[p.x, p.y, p.z] for p in landmarks]
                arr = np.array(points, dtype=np.float64).flatten()

                if hand_label == "Left":
                    left_hand = arr.tolist()
                elif hand_label == "Right":
                    right_hand = arr.tolist()

    return left_hand, right_hand


def process_dataset():
    """Process alphabet_fingerspelling dataset - organized by letter"""
    base_path = "data/alphabet_fingerspelling"
    output_path = "data/reference/fingerspelling_by_letter"

    os.makedirs(output_path, exist_ok=True)

    detector = create_detector()

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    folders = ["asl_alphabet_train", "asl_alphabet_test"]

    total_images = 0
    total_saved = 0

    for letter in ALPHABET:
        letter_output = os.path.join(output_path, letter)
        os.makedirs(letter_output, exist_ok=True)

        sample_count = 0

        for split in folders:
            folder_path = os.path.join(base_path, split, letter)
            if not os.path.exists(folder_path):
                continue

            # Get images
            images = glob(os.path.join(folder_path, "*.jpg"))

            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                lh, rh = extract_hand_landmarks(img, detector)

                # Use whichever hand has data
                if any(lh):
                    data = lh
                elif any(rh):
                    data = rh
                else:
                    continue

                # Save pickle
                sample_name = f"{letter}_{split}_{sample_count}"

                with open(os.path.join(letter_output, f"{sample_name}.pickle"), "wb") as f:
                    pkl.dump(data, f)

                sample_count += 1
                total_saved += 1

            total_images += len(images)

        if sample_count > 0:
            print(f"  {letter}: {sample_count} samples")

    detector.close()

    print(f"\nTotal: {total_saved} samples from {total_images} images")
    print(f"Saved to: {output_path}/")

    return total_saved


if __name__ == "__main__":
    process_dataset()
