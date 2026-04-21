#!/usr/bin/env python3
"""Train from alphabet_fingerspelling folder (Kaggle dataset)"""

import os
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/alphabet_fingerspelling"
OUTPUT_DIR = "data/reference/fingerspelling"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_features(image_path, detector):
    try:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
            # Use first detected hand
            landmarks = results.hand_landmarks[0]
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y])
            return np.array(features)
        return None
    except Exception as e:
        return None


def load_data():
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data dir not found: {DATA_DIR}")
        return None, None

    detector = create_detector()

    X = []
    y = []

    folders = [("asl_alphabet_train", "train"), ("asl_alphabet_test", "test")]

    for folder, split in folders:
        base_path = f"{DATA_DIR}/{folder}"
        if not os.path.exists(base_path):
            continue

        for letter_idx, letter in enumerate(ALPHABET):
            letter_path = f"{base_path}/{letter}"
            # Try different naming patterns
            images = (
                glob.glob(f"{letter_path}/*_{letter}*.jpg")
                + glob.glob(f"{letter_path}/*_{letter}*.jpeg")
                + glob.glob(f"{letter_path}/{letter}*.jpg")
                + glob.glob(f"{letter_path}/*train*.jpg")
                + glob.glob(f"{letter_path}/*test*.jpg")
            )
            images = list(set(images))  # Remove duplicates

            for img_path in images:
                features = extract_features(img_path, detector)
                if features is not None and len(features) == 42:
                    X.append(features)
                    y.append(letter_idx)

    detector.close()
    return np.array(X), np.array(y)


def train_model():
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading data...")
    X, y = load_data()

    if X is None or len(X) < 10:
        logger.error("Not enough data!")
        return

    logger.info(f"Loaded {len(X)} samples, {len(np.unique(y))} classes")
    logger.info(f"Label distribution: {np.unique(y, return_counts=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=True,
    )

    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    logger.info(f"Train accurate: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    joblib.dump((model, scaler), f"{OUTPUT_DIR}/model.joblib")
    logger.info(f"Model saved!")


if __name__ == "__main__":
    train_model()
