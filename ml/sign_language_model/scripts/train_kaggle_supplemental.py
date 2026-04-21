#!/usr/bin/env python3
"""Train Fingerspelling Model using Kaggle supplemental data"""

import os
import glob
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
OUTPUT_DIR = f"{DATA_DIR}/reference/fingerspelling"

# Target classes (digits 0-9 + letters a-z)
TARGET_DIGITS = list(range(15, 25))  # indices for "0"-"9"
TARGET_LETTERS = list(range(32, 58))  # indices for "a"-"z"

# Load character mapping
CHAR_MAP_FILE = f"{DATA_DIR}/downloads/character_to_prediction_index.json"
if os.path.exists(CHAR_MAP_FILE):
    with open(CHAR_MAP_FILE, "r") as f:
        CHAR_MAP = json.load(f)
    logger.info(f"Loaded char map with {len(CHAR_MAP)} entries")
else:
    CHAR_MAP = {}
    logger.warning("No character map found!")


def get_label_from_phrase(phrase):
    if not phrase:
        return None
    char = phrase[0].lower()
    if char in CHAR_MAP:
        return CHAR_MAP[char]
    return None


def load_data():
    all_features = []
    all_labels = []

    csv_path = f"{DATA_DIR}/downloads/supplemental_metadata.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} metadata rows")

    unique_files = df["file_id"].unique()
    logger.info(f"Found {len(unique_files)} unique files")

    for file_id in tqdm(unique_files[:30000], desc="Processing"):
        try:
            file_rows = df[df["file_id"] == file_id]
            npy_pattern = f"{DATA_DIR}/downloads/supplemental_landmarks/{file_id}/*.npy"
            npy_files = glob.glob(npy_pattern)

            if not npy_files:
                continue

            data = np.load(npy_files[0])

            for _, row in file_rows.iterrows():
                phrase = row["phrase"]
                label_idx = get_label_from_phrase(phrase)

                if label_idx is None:
                    continue

                if label_idx not in TARGET_DIGITS and label_idx not in TARGET_LETTERS:
                    continue

                frame = data[0]
                features = frame[:42]

                all_features.append(features)
                all_labels.append(label_idx)

        except Exception:
            continue

    return np.array(all_features), np.array(all_labels)


def train_model():
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading training data...")
    X, y = load_data()

    if len(X) < 100:
        logger.error("Not enough data!")
        return

    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"Unique labels: {np.unique(y)}")

    # Filter out NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    logger.info(f"After filtering NaN: {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=True,
    )

    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    logger.info(f"Train accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    model_path = f"{OUTPUT_DIR}/model.joblib"
    joblib.dump((model, scaler), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
