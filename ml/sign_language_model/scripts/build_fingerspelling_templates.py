#!/usr/bin/env python3
"""
Create reference templates for fingerspelling from images
"""

import os
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/raw_datasets/LSC70W/Per70"
OUTPUT_DIR = "data/reference/fingerspelling_templates"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image_path, detector):
    """Extract normalized hand landmarks from image"""
    try:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            # Extract x,y normalized (21 points)
            features = []
            for lm in hand[:21]:
                features.extend([lm.x, lm.y])
            return np.array(features)
        return None
    except Exception as e:
        return None


def build_templates():
    """Build average templates for each letter"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = create_detector()

    # Map folder names to letters
    # YO = Y, HOLA = H, NOMBRE = N, etc.
    folder_to_letter = {
        "YO": "Y",
        "HOLA": "H",
        "NOMBRE": "N",
        "BUENAS": "B",
        "DIAS": "D",
        "GUSTAR": "G",
        "TARDES": "T",
        "NOCHES": "N",
        "LICOR": "L",
        "ANNOS": "A",
    }

    templates = defaultdict(list)

    # Process all letter folders
    letter_folders = glob.glob(f"{DATA_DIR}/*/")

    for folder in letter_folders:
        folder_name = os.path.basename(folder)
        letter = folder_to_letter.get(folder_name)
        if not letter:
            continue

        images = glob.glob(f"{folder}/*.jpg")[:10]  # Use up to 10 images per letter

        for img_path in images:
            features = extract_hand_landmarks(img_path, detector)
            if features is not None:
                templates[letter].append(features)

    detector.close()

    # Save templates
    template_dict = {}
    for letter, features_list in templates.items():
        if features_list:
            # Store all samples, not just average
            template_dict[letter] = np.array(features_list)
            logger.info(f"{letter}: {len(features_list)} samples")

    # Save all templates
    np.save(f"{OUTPUT_DIR}/templates.npy", template_dict)
    logger.info(f"Saved {len(template_dict)} letter templates")

    return template_dict


def test_templates():
    """Test template matching"""
    templates_file = f"{OUTPUT_DIR}/templates.npy"
    if not os.path.exists(templates_file):
        logger.error("No templates found!")
        return

    templates = np.load(templates_file, allow_pickle=True).item()
    logger.info(f"Loaded templates for: {list(templates.keys())}")

    # Test with webcam
    import cv2
    from fastdtw import fastdtw

    detector = create_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Can't open camera!")
        return

    logger.info("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = detector.detect(mp_image)

        if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            features = np.array([[lm.x, lm.y] for lm in hand[:21]]).flatten()

            # Compare with all templates using DTW
            best_match = ""
            best_dist = float("inf")

            for letter, template_samples in templates.items():
                for template in template_samples:
                    dist, _ = fastdtw(features.reshape(1, -1), template.reshape(1, -1))
                    if dist < best_dist:
                        best_dist = dist
                        best_match = letter

            if best_match and best_dist < 50:  # threshold
                cv2.putText(
                    frame, f"{best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )

        cv2.imshow("Template Matching", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    build_templates()
