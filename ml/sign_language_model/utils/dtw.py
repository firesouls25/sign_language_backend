import pandas as pd
from fastdtw import fastdtw
import numpy as np
from models.core.sign_model import SignModel


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    print(
        f"DTW - Grabacion tiene {len(rec_left_hand)} frames izq, {len(rec_right_hand)} frames der"
    )

    for idx, row in reference_signs.iterrows():
        ref_sign_model = row["sign_model"]

        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            distance = 0.0

            if recorded_sign.has_left_hand and len(rec_left_hand) > 0 and len(ref_left_hand) > 0:
                dist_l, _ = fastdtw(rec_left_hand, ref_left_hand)
                distance += dist_l

            if recorded_sign.has_right_hand and len(rec_right_hand) > 0 and len(ref_right_hand) > 0:
                dist_r, _ = fastdtw(rec_right_hand, ref_right_hand)
                distance += dist_r

            reference_signs.at[idx, "distance"] = distance
        else:
            reference_signs.at[idx, "distance"] = np.inf

    return reference_signs.sort_values(by=["distance"])
