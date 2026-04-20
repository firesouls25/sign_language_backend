#!/usr/bin/env python3
"""
Train Fingerspelling Model using sklearn (same as handshape approach)
Works with the asl_keypoints data.
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "downloads", "asl_keypoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "fingerspelling")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load ASL keypoints dataset with error handling"""
    csv_path = os.path.join(DATA_DIR, "keypoint.csv")
    logger.info(f"Loading data from {csv_path}")

    data = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",") if x.strip()]
            if len(parts) >= 43:
                try:
                    data.append([float(x) for x in parts[:43]])
                except:
                    continue

    data = np.array(data)
    logger.info(f"Loaded data shape: {data.shape}")

    labels = data[:, 0].astype(int)
    features = data[:, 1:43]

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Unique labels: {sorted(np.unique(labels))}")

    return features, labels


def train_model(X, y):
    """Train MLP classifier"""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training MLP classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
    )

    model.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    return model, scaler, label_encoder


def save_model(model, scaler, label_encoder):
    """Save model and encoders"""
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")

    joblib.dump((model, scaler), model_path)
    np.save(labels_path, {label: idx for idx, label in enumerate(label_encoder.classes_)})

    logger.info(f"Model saved to {model_path}")


def main():
    logger.info("Starting fingerspelling model training...")

    if not os.path.exists(DATA_DIR):
        logger.error(f"Dataset not found at {DATA_DIR}")
        return

    X, y = load_data()

    logger.info(f"Training with {len(X)} samples...")
    model, scaler, label_encoder = train_model(X, y)
    save_model(model, scaler, label_encoder)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
