#!/usr/bin/env python3
"""
Train Handshape Model for ASL Signs
Uses the downloaded ASL image dataset with MediaPipe for landmark extraction.
"""

import os
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import glob
from collections import defaultdict
import joblib
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGN_LANGUAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(
    SIGN_LANGUAGE_MODEL_DIR, "data", "downloads", "processed_combine_asl_dataset"
)
OUTPUT_DIR = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "handshape")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_hand_detector():
    """Create MediaPipe hand landmark detector"""
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            SIGN_LANGUAGE_MODEL_DIR, "mediapipe/models/hand_landmarker.task"
        )
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(detector, image_path):
    """Extract hand landmarks from a single image"""
    try:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        left_landmarks = None
        right_landmarks = None

        if results and results.hand_landmarks:
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                handedness = (
                    results.handedness[i][0].category_name
                    if results.handedness and i < len(results.handedness)
                    else None
                )

                landmarks_array = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

                if handedness == "Left":
                    left_landmarks = landmarks_array
                elif handedness == "Right":
                    right_landmarks = landmarks_array

        return left_landmarks, right_landmarks
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None, None


def load_dataset():
    """Load ASL dataset and extract landmarks"""
    detector = create_hand_detector()

    all_landmarks = defaultdict(list)
    label_counts = defaultdict(int)

    letters = [chr(ord("a") + i) for i in range(26)]

    for letter in tqdm(letters, desc="Processing letters"):
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.exists(letter_dir):
            logger.warning(f"Directory not found: {letter_dir}")
            continue

        image_files = glob.glob(os.path.join(letter_dir, "*.jpg"))

        if len(image_files) < 5:
            logger.warning(f"Not enough images for letter {letter}: {len(image_files)}")
            continue

        for img_path in tqdm(image_files[:50], desc=f"  {letter}", leave=False):
            left, right = extract_landmarks_from_image(detector, img_path)

            if left or right:
                features = []
                left_flat = []
                right_flat = []

                if left and len(left) == 21:
                    for point in left:
                        left_flat.extend([point[0], point[1], point[2]])
                else:
                    left_flat = [0.0] * 63

                if right and len(right) == 21:
                    for point in right:
                        right_flat.extend([point[0], point[1], point[2]])
                else:
                    right_flat = [0.0] * 63

                features = left_flat + right_flat
                all_landmarks[letter].append(features)
                label_counts[letter] += 1

    detector.close()

    logger.info(f"Extracted landmarks for {len(all_landmarks)} letters")
    for letter, count in label_counts.items():
        logger.info(f"  {letter}: {count} samples")

    return all_landmarks, label_counts


def train_model(all_landmarks, label_counts):
    """Train MLP classifier for handshape recognition"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = []
    y = []

    for label, features_list in all_landmarks.items():
        for features in features_list:
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Labels: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    return model


def save_model(model, label_counts):
    """Save model and label encoder"""
    label_encoder = {label: idx for idx, label in enumerate(sorted(label_counts.keys()))}

    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")

    joblib.dump(model, model_path)
    np.save(labels_path, label_encoder)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Labels saved to {labels_path}")


def main():
    logger.info("Starting handshape model training...")

    if not os.path.exists(DATA_DIR):
        logger.error(f"Dataset not found at {DATA_DIR}")
        return

    all_landmarks, label_counts = load_dataset()

    if len(all_landmarks) < 10:
        logger.error("Not enough data to train model")
        return

    model = train_model(all_landmarks, label_counts)
    save_model(model, label_counts)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
