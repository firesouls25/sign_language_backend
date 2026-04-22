from typing import List, Optional

import numpy as np

from .hand_model import HandModel
from .pose_model import PoseModel


class SignModel:
    def __init__(
        self,
        left_hand_list: List[List[float]],
        right_hand_list: List[List[float]],
        pose_list: Optional[List[List[float]]] = None,
    ):
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)

        self.pose_list = pose_list if pose_list else []
        self.has_pose = len(self.pose_list) > 0 and np.sum(self.pose_list) != 0
        self.pose_embedding = self._get_pose_embedding_from_list(self.pose_list)

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

    @staticmethod
    def _get_pose_embedding_from_list(
        pose_list: List[List[float]],
    ) -> List[List[float]]:
        embedding = []
        for frame_idx in range(len(pose_list)):
            if np.sum(pose_list[frame_idx]) == 0:
                continue

            pose_model = PoseModel(pose_list[frame_idx])
            if pose_model.has_pose and pose_model.feature_vector:
                embedding.append(pose_model.feature_vector)
        return embedding

    def get_combined_embedding(self) -> np.ndarray:
        combined = []

        if self.lh_embedding:
            combined.append(np.mean(self.lh_embedding, axis=0))

        if self.rh_embedding:
            combined.append(np.mean(self.rh_embedding, axis=0))

        if self.pose_embedding:
            combined.append(np.mean(self.pose_embedding, axis=0))

        if not combined:
            return np.array([])

        return np.concatenate(combined)
