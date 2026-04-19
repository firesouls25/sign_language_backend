import numpy as np


class PoseModel:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    TORSO_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    LEFT_ARM_LANDMARKS = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
    RIGHT_ARM_LANDMARKS = [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]

    def __init__(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            self.feature_vector = []
            self.has_pose = False
            return

        landmarks = np.array(landmarks).reshape((-1, 3))

        self.has_pose = len(landmarks) >= 25

        if not self.has_pose:
            self.feature_vector = []
            return

        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> list:
        torso_center = self._get_torso_center(landmarks)
        torso_size = self._get_torso_size(landmarks)

        if torso_size == 0:
            torso_size = 1

        normalized_left_arm = self._normalize_limb(
            landmarks, self.LEFT_ARM_LANDMARKS, torso_center, torso_size
        )
        normalized_right_arm = self._normalize_limb(
            landmarks, self.RIGHT_ARM_LANDMARKS, torso_center, torso_size
        )

        normalized_torso = self._normalize_torso(landmarks, torso_center, torso_size)

        left_arm_angles = self._get_angles_from_connections(normalized_left_arm)
        right_arm_angles = self._get_angles_from_connections(normalized_right_arm)
        torso_angles = self._get_torso_angles(normalized_torso)

        return left_arm_angles + right_arm_angles + torso_angles

    def _get_torso_center(self, landmarks: np.ndarray) -> np.ndarray:
        shoulder_center = (landmarks[self.LEFT_SHOULDER] + landmarks[self.RIGHT_SHOULDER]) / 2
        hip_center = (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP]) / 2
        return (shoulder_center + hip_center) / 2

    def _get_torso_size(self, landmarks: np.ndarray) -> float:
        shoulder_width = np.linalg.norm(
            landmarks[self.LEFT_SHOULDER] - landmarks[self.RIGHT_SHOULDER]
        )
        hip_width = np.linalg.norm(landmarks[self.LEFT_HIP] - landmarks[self.RIGHT_HIP])
        torso_height = np.linalg.norm(
            (landmarks[self.LEFT_SHOULDER] + landmarks[self.RIGHT_SHOULDER]) / 2
            - (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP]) / 2
        )
        return (shoulder_width + hip_width + torso_height) / 3

    def _normalize_limb(
        self, landmarks: np.ndarray, indices: list, center: np.ndarray, scale: float
    ) -> np.ndarray:
        limb_points = np.array([landmarks[i] for i in indices])
        limb_points = (limb_points - center) / scale
        return limb_points

    def _normalize_torso(
        self, landmarks: np.ndarray, center: np.ndarray, scale: float
    ) -> np.ndarray:
        torso_points = np.array([landmarks[i] for i in self.TORSO_LANDMARKS])
        torso_points = (torso_points - center) / scale
        return torso_points

    def _get_angles_from_connections(self, points: np.ndarray) -> list:
        angles = []
        connections = [(0, 1), (1, 2)]

        for i, (start, end) in enumerate(connections):
            if i + 1 < len(connections):
                v1 = points[end] - points[start]
                next_start, next_end = connections[i + 1]
                v2 = points[next_end] - points[next_start]
                angle = self._get_angle_between_vectors(v1, v2)
                angles.append(angle)

        return angles

    def _get_torso_angles(self, torso_points: np.ndarray) -> list:
        angles = []

        connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
        ]

        for start, end in connections:
            v1 = torso_points[end] - torso_points[start]
            for start2, end2 in connections:
                if (start, end) != (start2, end2):
                    v2 = torso_points[end2] - torso_points[start2]
                    angle = self._get_angle_between_vectors(v1, v2)
                    angles.append(angle)

        return angles[:6]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        if np.array_equal(u, v):
            return 0.0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return 0.0
        cos_angle = dot_product / norm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))
