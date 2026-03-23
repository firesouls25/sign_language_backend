from typing import List

import numpy as np

from models.core.hand_model import HandModel


class SignModel:
    def __init__(self, left_hand_list: List[List[float]], right_hand_list: List[List[float]]):
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)

    @staticmethod
    def _get_embedding_from_landmark_list(
        hand_list: List[List[float]],
    ) -> List[List[float]]:
        embedding = []
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
        return embedding
