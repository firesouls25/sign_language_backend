#!/usr/bin/env python3
"""
Try to detect landmarks from asl_landmarks images - they may have colored dots
or can be parsed as landmark positions
"""

import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_detector():
    base_options = python.BaseOptions(model_asset_path="mediapipe/models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def find_colored_dots(image_path):
    """Try to find colored dots that might be landmarks"""
    img = Image.open(image_path)
    arr = np.array(img)

    h, w = arr.shape[:2]

    # Look for non-background pixels (colored dots)
    # Background appears to be mostly white or black
    mask = (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)

    # Find coordinates of colored pixels
    coords = np.where(mask)

    if len(coords[0]) == 0:
        return []

    # Cluster nearby points - take approximate centers of clusters
    # Simplified: just get bounding area of all dots
    y_coords, x_coords = coords

    # Create a simple feature: normalized center + spread
    features = []

    if len(x_coords) > 0:
        # Normalize coordinates
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()

        # Calculate normalized positions for a simplified 21-point estimate
        # This is approximate - we don't know exact landmark positions

        # Center of mass
        center_x = x_coords.mean() / w
        center_y = y_coords.mean() / h

        features = [center_x, center_y, 0.0] * 21  # Fill with center

    return features


# Try processing both datasets
detector = create_detector()

datasets = [
    ("asl_landmarks", "data/downloads/asl_landmarks/Data"),
    ("processed_combine", "data/downloads/processed_combine_asl_dataset"),
]

results = {}

for name, base_path in datasets:
    print(f"\n=== {name} ===")

    if name == "asl_landmarks":
        # Check first image
        sample = glob(f"{base_path}/A/*.jpg")[0]
    else:
        sample = glob(f"{base_path}/a/*.jpg")[0]

    print(f"Testing: {sample}")

    # Try MediaPipe
    img_cv = cv2.imread(sample)
    if img_cv is not None:
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_results = detector.detect(mp_image)
        hands = len(mp_results.hand_landmarks) if mp_results.hand_landmarks else 0
        print(f"  MediaPipe: {hands} hands")

    # Try colored dots detection
    dots = find_colored_dots(sample)
    print(f"  Colored dots: {len(dots) // 3 if dots else 0} approx features")

detector.close()
print("\nDone")
