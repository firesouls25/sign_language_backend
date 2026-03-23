import os
import numpy as np
import pickle as pkl
import pandas as pd
from glob import glob
from tqdm import tqdm

from models.core.sign_model import SignModel


def load_dataset_lsc70():
    dataset_path = "data/dataset"

    if not os.path.exists(dataset_path):
        return []

    signs = []
    for sign_folder in os.listdir(dataset_path):
        sign_path = os.path.join(dataset_path, sign_folder)
        if not os.path.isdir(sign_path):
            continue

        for lh_file in glob(os.path.join(sign_path, "lh_*.pickle")):
            sample_name = os.path.basename(lh_file).replace("lh_", "").replace(".pickle", "")
            signs.append(sample_name)

    return signs


def load_reference_signs_lsc70():
    dataset_path = "data/dataset"
    reference_signs = {"name": [], "sign_model": [], "distance": []}

    for sign_folder in os.listdir(dataset_path):
        sign_path = os.path.join(dataset_path, sign_folder)
        if not os.path.isdir(sign_path):
            continue

        for lh_file in glob(os.path.join(sign_path, "lh_*.pickle")):
            sample_name = os.path.basename(lh_file).replace("lh_", "").replace(".pickle", "")
            sign_name = sample_name.split("_")[0]

            with open(lh_file, "rb") as f:
                left_hand_list = pkl.load(f)
            with open(lh_file.replace("lh_", "rh_"), "rb") as f:
                right_hand_list = pkl.load(f)

            left_hand_list = [
                np.array(x).flatten().tolist() if isinstance(x, np.ndarray) else x
                for x in left_hand_list
            ]
            right_hand_list = [
                np.array(x).flatten().tolist() if isinstance(x, np.ndarray) else x
                for x in right_hand_list
            ]

            reference_signs["name"].append(sign_name)
            reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
            reference_signs["distance"].append(0.0)

    df = pd.DataFrame(reference_signs, dtype=object)
    print(f"\nDataset LSC70 - {len(df)} muestras:")
    print(df.groupby("name").size().to_string())

    return df
