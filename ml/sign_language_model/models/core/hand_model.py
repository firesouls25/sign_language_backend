from typing import List

import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections


class HandModel:
    def __init__(self, landmarks):
        self.connections = HandLandmarksConnections.HAND_CONNECTIONS

        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for connection_from in connections:
            for connection_to in connections:
                angle = self._get_angle_between_vectors(connection_from, connection_to)
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    def _get_connections_from_landmarks(self, landmarks: np.ndarray) -> List[np.ndarray]:
        return [landmarks[conn.end] - landmarks[conn.start] for conn in self.connections]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
