#!/usr/bin/env python3
"""
Extract hand landmarks from fingerspelling images
Saves as pickle files for DTW comparison
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


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image, hands_model):
    """Extract hand landmarks, return left and right hand arrays (63 values each)"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = hands_model.detect(mp_image)

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


def process_fingerspelling_dataset():
    """Process alphabet_fingerspelling dataset"""
    base_path = "data/alphabet_fingerspelling"
    output_path = "data/reference/fingerspelling_pickles"

    os.makedirs(output_path, exist_ok=True)

    detector = create_detector()

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Process train and test folders
    folders = ["asl_alphabet_train", "asl_alphabet_test"]

    total_processed = 0
    errors = 0

    for split in folders:
        for letter in ALPHABET:
            letter_folder = os.path.join(base_path, split, letter)
            if not os.path.exists(letter_folder):
                continue

            # Find all images
            images = glob(os.path.join(letter_folder, "*.jpg")) + glob(
                os.path.join(letter_folder, "*.jpeg")
            )

            if not images:
                continue

            left_hand_list = []
            right_hand_list = []

            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    errors += 1
                    continue

                lh, rh = extract_hand_landmarks(img, detector)
                left_hand_list.append(lh)
                right_hand_list.append(rh)

            if len(left_hand_list) > 0:
                # Save as pickle
                sample_name = f"{letter}_{split}"

                with open(os.path.join(output_path, f"lh_{sample_name}.pickle"), "wb") as f:
                    pkl.dump(left_hand_list, f)
                with open(os.path.join(output_path, f"rh_{sample_name}.pickle"), "wb") as f:
                    pkl.dump(right_hand_list, f)

                print(f"  {sample_name}: {len(left_hand_list)} samples")
                total_processed += 1

    detector.close()

    print(f"\nCompleted: {total_processed} samples, {errors} errors")
    return total_processed, errors


if __name__ == "__main__":
    process_fingerspelling_dataset()
