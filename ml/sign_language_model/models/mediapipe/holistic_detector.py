from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HolisticResults:
    left_hand_landmarks: Optional[list] = None
    right_hand_landmarks: Optional[list] = None
    pose_landmarks: Optional[list] = None


class MediapipeHolisticDetector:
    def __init__(
        self,
        hand_model_path: str = "mediapipe/models/hand_landmarker.task",
        pose_model_path: str = "mediapipe/models/pose_landmarker.task",
        num_hands: int = 2,
        running_mode: str = "VIDEO",
        static_image_mode: bool = False,
    ):
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path
        self.num_hands = num_hands
        self.running_mode = self._get_running_mode(running_mode, static_image_mode)
        self._hand_detector = self._create_hand_detector()
        self._pose_detector = self._create_pose_detector()

    def _get_running_mode(self, running_mode: str, static_image_mode: bool):
        if static_image_mode:
            return vision.RunningMode.IMAGE
        modes = {
            "VIDEO": vision.RunningMode.VIDEO,
            "LIVE_STREAM": vision.RunningMode.LIVE_STREAM,
        }
        return modes.get(running_mode.upper(), vision.RunningMode.VIDEO)

    def _create_hand_detector(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _create_pose_detector(self):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=self.running_mode,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def detect(self, image, timestamp_ms: int = 0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self.running_mode == vision.RunningMode.VIDEO:
            hand_results = self._hand_detector.detect_for_video(mp_image, timestamp_ms)
            pose_results = self._pose_detector.detect_for_video(mp_image, timestamp_ms)
        elif self.running_mode == vision.RunningMode.IMAGE:
            hand_results = self._hand_detector.detect(mp_image)
            pose_results = self._pose_detector.detect(mp_image)
        else:
            hand_results = self._hand_detector.detect(mp_image)
            pose_results = self._pose_detector.detect(mp_image)

        return self._parse_results(hand_results, pose_results)

    def _parse_results(self, hand_results, pose_results) -> HolisticResults:
        results = HolisticResults()

        if hand_results and len(hand_results.hand_landmarks) > 0:
            handedness = hand_results.handedness
            for idx in range(len(hand_results.hand_landmarks)):
                hand_label = handedness[idx][0].category_name
                if hand_label == "Left":
                    results.left_hand_landmarks = hand_results.hand_landmarks[idx]
                elif hand_label == "Right":
                    results.right_hand_landmarks = hand_results.hand_landmarks[idx]

        if pose_results and pose_results.pose_landmarks:
            results.pose_landmarks = pose_results.pose_landmarks[0]

        return results

    def close(self):
        if hasattr(self._hand_detector, "close"):
            self._hand_detector.close()
        if hasattr(self._pose_detector, "close"):
            self._pose_detector.close()
