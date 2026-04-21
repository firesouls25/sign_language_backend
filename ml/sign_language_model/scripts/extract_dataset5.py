#!/usr/bin/env python3
"""
Extract landmarks from dataset5 (Kinect data) - LIMITED VERSION
"""

import cv2
import os
import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path, detector):
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results and results.hand_landmarks:
        for idx, hand_classification in enumerate(results.handedness):
            if len(hand_classification) > 0:
                hand_label = hand_classification[0].category_name
                landmarks = results.hand_landmarks[idx]
                points = [[p.x, p.y, p.z] for p in landmarks]
                return np.array(points, dtype=np.float64).flatten().tolist()

    return None


def process_dataset5():
    base_path = "data/dataset5"
    output_path = "data/reference/fingerspelling_by_letter_v2"

    os.makedirs(output_path, exist_ok=True)
    detector = create_detector()

    letters = list("abcdefghiklmnopqrstuvwxy")
    total_saved = 0

    for letter in letters:
        # Get images - limit total to avoid timeout
        images = glob.glob(os.path.join(base_path, f"*/{letter}/*.png"))[:300]
        if not images:
            continue

        letter_upper = letter.upper()
        letter_output = os.path.join(output_path, letter_upper)
        os.makedirs(letter_output, exist_ok=True)

        sample_count = 0
        for img_path in images:
            data = extract_landmarks(img_path, detector)
            if data is None:
                continue

            with open(
                os.path.join(letter_output, f"{letter_upper}_{sample_count}.pickle"), "wb"
            ) as f:
                pkl.dump(data, f)

            sample_count += 1
            total_saved += 1

        print(f"  {letter_upper}: {sample_count} samples")

    detector.close()
    print(f"\nTotal: {total_saved} samples")


if __name__ == "__main__":
    process_dataset5()
