import cv2
import os
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


def extract_hand_landmarks(image, hands_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = hands_model.detect(mp_image)

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if results and len(results.hand_landmarks) > 0:
        for idx, handedness in enumerate(results.handedness):
            hand_label = handedness[0].category_name
            landmarks = results.hand_landmarks[idx]
            points = [[p.x, p.y, p.z] for p in landmarks]
            arr = np.array(points, dtype=np.float64).flatten()
            if hand_label == "Left":
                left_hand = arr.tolist()
            elif hand_label == "Right":
                right_hand = arr.tolist()

    return left_hand, right_hand


def process_lsc70_dataset():
    base_path = "data/raw_datasets/LSC70/LSC70W"
    output_path = "data/dataset"

    os.makedirs(output_path, exist_ok=True)

    hands_model = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path="mediapipe/models/hand_landmarker.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )

    sign_categories = [
        "HOLA",
        "NOMBRE",
        "BUENAS",
        "TARDES",
        "NOCHES",
        "DIAS",
        "GUSTAR",
        "LICOR",
        "YO",
        "ANNOS",
    ]

    person_folders = sorted([d for d in os.listdir(base_path) if d.startswith("Per")])

    total_processed = 0
    errors = 0

    print(f"Procesando {len(person_folders)} personas, {len(sign_categories)} señas...")

    for person in tqdm(person_folders, desc="Personas"):
        for sign in sign_categories:
            sign_folder = os.path.join(base_path, person, sign)
            if not os.path.exists(sign_folder):
                continue

            image_files = sorted(glob(os.path.join(sign_folder, "*.jpg")))
            if not image_files:
                continue

            left_hand_list = []
            right_hand_list = []

            for img_path in image_files:
                img = cv2.imread(img_path)
                if img is None:
                    errors += 1
                    continue

                lh, rh = extract_hand_landmarks(img, hands_model)
                left_hand_list.append(lh)
                right_hand_list.append(rh)

            if len(left_hand_list) > 0:
                sample_name = f"{sign}_{person}"

                sign_output = os.path.join(output_path, sign)
                os.makedirs(sign_output, exist_ok=True)

                with open(
                    os.path.join(sign_output, f"lh_{sample_name}.pickle"), "wb"
                ) as f:
                    pkl.dump(left_hand_list, f)
                with open(
                    os.path.join(sign_output, f"rh_{sample_name}.pickle"), "wb"
                ) as f:
                    pkl.dump(right_hand_list, f)
                with open(
                    os.path.join(sign_output, f"pose_{sample_name}.pickle"), "wb"
                ) as f:
                    pkl.dump([[0.0] * 99 for _ in left_hand_list], f)

                total_processed += 1

    print(f"\nProcesamiento completado:")
    print(f"  - Muestras procesadas: {total_processed}")
    print(f"  - Errores: {errors}")

    return total_processed, errors


if __name__ == "__main__":
    process_lsc70_dataset()
