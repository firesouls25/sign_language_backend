#!/usr/bin/env python3
"""
Train Fingerspelling Model from pre-extracted keypoints CSV.
Uses existing asl_keypoints/keypoint.csv data.
"""

import os
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGN_LANGUAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(
    SIGN_LANGUAGE_MODEL_DIR, "data", "downloads", "asl_keypoints", "keypoint.csv"
)
OUTPUT_DIR = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "fingerspelling")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
ALL_LABELS = ALPHABET + DIGITS
NUM_CLASSES = len(ALL_LABELS)

LABEL_MAP = {i: label for i, label in enumerate(ALL_LABELS)}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}


def load_csv_data():
    """Load keypoints from CSV"""
    logger.info(f"Loading data from {DATA_DIR}")

    data = np.loadtxt(DATA_DIR, delimiter=",")
    logger.info(f"Loaded data shape: {data.shape}")

    labels = data[:, 0].astype(int)
    features = data[:, 1:]

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Unique labels: {np.unique(labels)}")

    return features, labels


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


def train_model(X, y):
    """Train TensorFlow model"""
    from sklearn.model_selection import train_test_split

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Number of classes: {len(np.unique(y))}")

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


def save_model(model):
    """Save model and label map"""
    model_path = os.path.join(OUTPUT_DIR, "custom_fingerspelling.keras")

    model.save(model_path)

    logger.info(f"Model saved to {model_path}")


def main():
    logger.info("Starting fingerspelling model training from CSV...")
    logger.info(f"Labels: {ALL_LABELS} ({NUM_CLASSES} classes)")

    if not os.path.exists(DATA_DIR):
        logger.error(f"Data not found at {DATA_DIR}")
        return

    X, y = load_csv_data()

    total_samples = len(y)
    logger.info(f"Total samples: {total_samples}")

    model = train_model(X, y)
    save_model(model)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
