#!/usr/bin/env python3
"""
Train Fingerspelling Model for ASL Letters and Digits
Uses processed_combine_asl_dataset with MediaPipe for landmark extraction.
Trains a TensorFlow model for fingerspelling (A-Z + 0-9 = 36 classes).
"""

import os
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import glob
from collections import defaultdict
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGN_LANGUAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "raw_datasets", "LSC70W", "Per70")
OUTPUT_DIR = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "fingerspelling")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
ALL_LABELS = ALPHABET + DIGITS
NUM_CLASSES = len(ALL_LABELS)

LABEL_MAP = {i: label for i, label in enumerate(ALL_LABELS)}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}

FOLDER_TO_LETTER = {
    "YO": "y",
    "HOLA": "h",
    "NOMBRE": "n",
    "BUENAS": "b",
    "DIAS": "d",
    "GUSTAR": "g",
    "TARDES": "t",
    "NOCHES": "n",
    "LICOR": "l",
    "ANNOS": "a",
}


def create_hand_detector():
    """Create MediaPipe hand landmark detector for images"""
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            SIGN_LANGUAGE_MODEL_DIR, "mediapipe/models/hand_landmarker.task"
        )
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(detector, image_path):
    """Extract hand landmarks from a single image"""
    try:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        landmarks = None

        if results and results.hand_landmarks:
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                landmarks_array = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                if len(landmarks_array) == 21:
                    landmarks = landmarks_array
                    break

        return landmarks
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def load_dataset(max_per_label=500):
    """Load ASL dataset and extract landmarks"""
    detector = create_hand_detector()

    all_landmarks = defaultdict(list)
    label_counts = defaultdict(int)

    for label in tqdm(ALL_LABELS, desc="Processing labels"):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            logger.warning(f"Directory not found: {label_dir}")
            continue

        image_files = (
            glob.glob(os.path.join(label_dir, "*.jpg"))
            + glob.glob(os.path.join(label_dir, "*.jpeg"))
            + glob.glob(os.path.join(label_dir, "*.png"))
        )

        if len(image_files) == 0:
            logger.warning(f"No images for label {label}")
            continue

        random.shuffle(image_files)
        num_to_process = min(len(image_files), max_per_label)

        for img_path in tqdm(image_files[:num_to_process], desc=f"  {label}", leave=False):
            landmarks = extract_landmarks_from_image(detector, img_path)

            if landmarks and len(landmarks) == 21:
                features = []
                for point in landmarks:
                    features.extend([point[0], point[1], point[2]])

                all_landmarks[label].append(features)
                label_counts[label] += 1

    detector.close()

    logger.info(f"Extracted landmarks for {len(all_landmarks)} labels")
    for label in ALL_LABELS:
        if label_counts[label] > 0:
            logger.info(f"  {label}: {label_counts[label]} samples")

    return all_landmarks, label_counts


def build_model(input_dim=63, num_classes=36):
    """Build TensorFlow model for fingerspelling"""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(all_landmarks, label_counts):
    """Train TensorFlow model"""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    X = []
    y = []

    for label, features_list in all_landmarks.items():
        label_idx = LABEL_TO_IDX[label]
        for features in features_list:
            X.append(features)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Number of classes: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_model(input_dim=X.shape[1], num_classes=NUM_CLASSES)

    logger.info("Training model...")
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    return model


def save_model(model, label_counts):
    """Save model and label map"""
    model_path = os.path.join(OUTPUT_DIR, "custom_fingerspelling.keras")
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.npy")

    model.save(model_path)
    np.save(label_map_path, LABEL_MAP)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Label map saved to {label_map_path}")


def main():
    logger.info("Starting fingerspelling model training...")
    logger.info(f"Labels: {ALL_LABELS} ({NUM_CLASSES} classes)")

    if not os.path.exists(DATA_DIR):
        logger.error(f"Dataset not found at {DATA_DIR}")
        return

    max_per_label = 500
    logger.info(f"Processing max {max_per_label} images per label...")

    all_landmarks, label_counts = load_dataset(max_per_label=max_per_label)

    if len(all_landmarks) < 10:
        logger.error("Not enough data to train model")
        return

    total_samples = sum(label_counts.values())
    logger.info(f"Total samples: {total_samples}")

    model = train_model(all_landmarks, label_counts)
    save_model(model, label_counts)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
