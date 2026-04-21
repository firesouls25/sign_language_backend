#!/usr/bin/env python3
"""
Train Fingerspelling Model using Kaggle dataset
Uses supplemental_landmarks with letters (a-z) and digits (0-9)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/downloads"
OUTPUT_DIR = "data/reference/fingerspelling"

# Load character mapping
with open(f"{DATA_DIR}/character_to_prediction_index.json", "r") as f:
    CHAR_MAP = json.load(f)

# Create reverse mapping: index -> character
IDX_TO_CHAR = {v: k for k, v in CHAR_MAP.items()}

# Define which classes we want (letters a-z + digits 0-9)
TARGET_DIGITS = list(range(15, 25))  # indices 15-24 = "0"-"9"
TARGET_LETTERS = list(range(32, 58))  # indices 32-57 = "a"-"z"
TARGET_CLASSES = TARGET_DIGITS + TARGET_LETTERS

logger.info(f"Character map sample: {list(CHAR_MAP.items())[:10]}")
logger.info(f"Target indices: digits={TARGET_DIGITS}, letters={TARGET_LETTERS}")


def load_training_data(max_files=50000):
    """Load training data from supplemental_landmarks"""
    all_features = []
    all_labels = []

    # Read metadata to get phrase -> label mapping
    df = pd.read_csv(f"{DATA_DIR}/supplemental_metadata.csv")
    logger.info(f"Loaded {len(df)} metadata rows")

    # Group by file_id to find unique sequences
    unique_files = df["file_id"].unique()
    logger.info(f"Found {len(unique_files)} unique files")

    # Load each file
    files_to_load = min(len(unique_files), max_files)

    for idx, file_id in enumerate(tqdm(unique_files[:files_to_load], desc="Loading files")):
        try:
            # Get all rows for this file_id
            file_rows = df[df["file_id"] == file_id]

            for _, row in file_rows.iterrows():
                phrase = row["phrase"]
                path = row["path"]

                # Find matching npy files
                npy_dir = path.replace(".parquet", "").replace("supplemental_landmarks/", "")
                npy_pattern = f"{DATA_DIR}/supplemental_landmarks/{npy_dir}/*.npy"

                npy_files = glob.glob(npy_pattern)
                if not npy_files:
                    continue

                # Load npy (shape: frames x 390)
                try:
                    data = np.load(npy_files[0])
                except:
                    continue

                # Select first frame for static prediction
                features = data[0]  # shape: (390,)

                # Extract x,y for each of 21 hand points (indices 0,1 for x,y of point 0, etc.)
                # Format in data: x1_lh, y1_lh, x1_rh, y1_rh, ... or similar
                # We need first 42 values (21 points x,y)
                hand_features = features[:42]

                # Get the label character from phrase (first char)
                if len(phrase) > 0:
                    char = phrase[0].lower()
                    if char in CHAR_MAP:
                        label = CHAR_MAP[char]
                        if label in TARGET_CLASSES:
                            all_features.append(hand_features)
                            all_labels.append(label)

        except Exception as e:
            continue

    return np.array(all_features), np.array(all_labels)


def build_model(input_dim=42, num_classes=36):
    """Build sklearn MLP model"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

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
    return model


def train_and_save():
    """Main training function"""
    import joblib

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    X, y = load_training_data(max_files=30000)

    if len(X) < 100:
        logger.error("Not enough training data!")
        return

    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"Label distribution: {np.unique(y, return_counts=True)}")

    # Split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    logger.info("Training model...")
    model = build_model(input_dim=X.shape[1])
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    logger.info(f"Train accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save
    model_path = f"{OUTPUT_DIR}/model.joblib"
    joblib.dump((model, scaler), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save()
