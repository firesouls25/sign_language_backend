import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class KeypointExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("KeypointExtractor initialized")

    def extract_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        keypoints = {
            "hands": [],
            "pose": None,
            "face": None
        }
        
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                keypoints["hands"].append([
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in hand_landmarks.landmark
                ])
        
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            keypoints["pose"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in pose_results.pose_landmarks.landmark
            ]
        
        face_results = self.face.process(rgb_frame)
        if face_results.multi_face_landmarks:
            keypoints["face"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in face_results.multi_face_landmarks[0].landmark
            ]
        
        return keypoints

    def close(self):
        self.hands.close()
        self.pose.close()
        self.face.close()


class SignRecognizer:
    def __init__(self):
        self.keypoint_extractor = KeypointExtractor()
        self.sequence_buffer: List[Dict] = []
        self.sequence_length = 30
        logger.info("SignRecognizer initialized")

    def process_frame(self, frame: np.ndarray) -> Dict:
        keypoints = self.keypoint_extractor.extract_keypoints(frame)
        
        if keypoints:
            self.sequence_buffer.append(keypoints)
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)
        
        result = self._recognize_sign()
        
        return {
            "keypoints": keypoints,
            "text": result["text"],
            "confidence": result["confidence"],
            "sequence_length": len(self.sequence_buffer)
        }

    def _recognize_sign(self) -> Dict:
        if len(self.sequence_buffer) < 5:
            return {"text": "", "confidence": 0.0}
        
        return {
            "text": "Signs detected",
            "confidence": 0.75
        }

    def reset_sequence(self):
        self.sequence_buffer = []

    def close(self):
        self.keypoint_extractor.close()


keypoint_extractor = None
sign_recognizer = None


def get_keypoint_extractor() -> KeypointExtractor:
    global keypoint_extractor
    if keypoint_extractor is None:
        keypoint_extractor = KeypointExtractor()
    return keypoint_extractor


def get_sign_recognizer() -> SignRecognizer:
    global sign_recognizer
    if sign_recognizer is None:
        sign_recognizer = SignRecognizer()
    return sign_recognizer
