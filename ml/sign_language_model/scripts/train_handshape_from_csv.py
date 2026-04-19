#!/usr/bin/env python3
"""
Train Handshape Model from ASL Keypoints CSV Dataset
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "downloads", "asl_keypoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "handshape")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SPANISH_MAPPING = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "del": "DEL",
    "nothing": "NADA",
    "space": "ESPACIO",
}


def load_data():
    """Load ASL keypoints dataset"""
    csv_path = os.path.join(DATA_DIR, "keypoint.csv")
    logger.info(f"Loading data from {csv_path}")

    # Try to read with error handling for inconsistent columns
    try:
        df = pd.read_csv(csv_path, header=None, on_bad_lines="skip")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        # Fallback: read and process line by line
        data = []
        with open(csv_path, "r") as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",") if x.strip()]
                if len(parts) >= 43:
                    try:
                        data.append([float(x) for x in parts[:43]])
                    except:
                        continue
        df = pd.DataFrame(data)

    logger.info(f"Dataset shape after cleanup: {df.shape}")
    logger.info(f"First few rows:\n{df.head()}")

    labels = df.iloc[:, 0].astype(int)
    features = df.iloc[:, 1:43].values  # Take only first 42 features (21 points * 2 coords)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Unique labels: {sorted(labels.unique())}")

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

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    y_pred = model.predict(X_test_scaled)
    class_names = [str(c) for c in label_encoder.classes_]
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))

    return model, scaler, label_encoder


def save_model(model, scaler, label_encoder):
    """Save model and encoders"""
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")

    joblib.dump((model, scaler), model_path)
    np.save(labels_path, {label: idx for idx, label in enumerate(label_encoder.classes_)})

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Labels saved to {labels_path}")


def main():
    logger.info("Starting handshape model training...")

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
