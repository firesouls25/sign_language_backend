#!/usr/bin/env python3
"""
Train Fingerspelling Model - Uses train_landmarks_npy data
Requires train.csv for labels. If not available, tries to use supplemental data.
"""

import os
import glob
import numpy as np
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
OUTPUT_DIR = f"{DATA_DIR}/reference/fingerspelling"

# Load character mapping if available
CHAR_MAP = {}
if os.path.exists(f"{DATA_DIR}/downloads/character_to_prediction_index.json"):
    with open(f"{DATA_DIR}/downloads/character_to_prediction_index.json", "r") as f:
        CHAR_MAP = json.load(f)

logger.info(f"Character map: {len(CHAR_MAP)} entries")
logger.info(f"Sample: {list(CHAR_MAP.items())[:5]}")


def train_model():
    """Extract features and labels from train_landmarks_npy files"""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading train_landmarks_npy...")

    # Find all npy files
    npy_files = glob.glob(f"{DATA_DIR}/train_landmarks_npy/*/*.npy")
    logger.info(f"Found {len(npy_files)} files")

    if len(npy_files) == 0:
        logger.error("No data found!")
        return

    # Load a sample to check dimensions
    sample = np.load(npy_files[0])
    logger.info(f"Sample shape: {sample.shape}")  # (frames, 390)

    # Extract features: first frame, first 42 values (x,y for 21 hand points)
    X = []
    y = []

    TARGET_DIGITS = list(range(15, 25))  # "0"-"9"
    TARGET_LETTERS = list(range(32, 58))  # "a"-"z"

    for npy_file in tqdm(npy_files[:50000], desc="Processing"):
        try:
            data = np.load(npy_file)
            # Use first frame
            frame = data[0]

            # Extract hand landmarks (21 points x 2 = 42 features)
            # Looking at the data format: first values seem to be x,y coordinates
            features = frame[:42]

            # Try to get label from filename
            # Files named like: train_landmarks_npy/{participant}/{sequence_id}.npy
            # No direct label mapping available

            # For now, use the features directly without labels
            # We'll train an autoencoder or use clustering

            X.append(features)

        except Exception as e:
            continue

    X = np.array(X)
    logger.info(f"Loaded {len(X)} samples")

    if len(X) < 100:
        logger.error("Not enough data!")
        return

    logger.info(f"X shape: {X.shape}")

    # Since we don't have labels, train with all data as one class
    # Or we can try to find labels from supplemental data
    # For now, create a simple model that recognizes the patterns

    # Use self-supervised: just train to reconstruct
    # Then use the reconstruction error for detection

    logger.info("Training model...")

    # Simply scale and return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # For now, just save the scaler and data stats
    # A real classifier needs labels
    logger.info("Model needs labels - will save scaler only")

    joblib.dump({"scaler": scaler, "n_samples": len(X)}, f"{OUTPUT_DIR}/stats.joblib")
    logger.info(f"Saved to {OUTPUT_DIR}/stats.joblib")


if __name__ == "__main__":
    train_model()
