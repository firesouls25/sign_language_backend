from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HandLandmarks:
    landmarks: List
    handedness: str
    confidence: float


class MediapipeHandDetector:
    def __init__(
        self,
        model_path: str = "mediapipe/models/hand_landmarker.task",
        num_hands: int = 2,
        running_mode: str = "VIDEO",
        static_image_mode: bool = False,
    ):
        self.model_path = model_path
        self.num_hands = num_hands
        self.running_mode = self._get_running_mode(running_mode, static_image_mode)
        self._detector = self._create_detector()

    def _get_running_mode(self, running_mode: str, static_image_mode: bool):
        if static_image_mode:
            return vision.RunningMode.IMAGE
        modes = {
            "VIDEO": vision.RunningMode.VIDEO,
            "LIVE_STREAM": vision.RunningMode.LIVE_STREAM,
        }
        return modes.get(running_mode.upper(), vision.RunningMode.VIDEO)

    def _create_detector(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
        )
        return vision.HandLandmarker.create_from_options(options)

    def detect(self, image, timestamp_ms: int = 0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self.running_mode == vision.RunningMode.VIDEO:
            results = self._detector.detect_for_video(mp_image, timestamp_ms)
        elif self.running_mode == vision.RunningMode.IMAGE:
            results = self._detector.detect(mp_image)
        else:
            results = self._detector.detect(mp_image)

        return self._parse_results(results)

    def _parse_results(self, results) -> dict:
        parsed = {
            "left_hand": None,
            "right_hand": None,
            "all_landmarks": [],
        }

        if results is None or not hasattr(results, "hand_landmarks"):
            return parsed

        if len(results.hand_landmarks) > 0:
            handedness = results.handedness
            for idx in range(len(results.hand_landmarks)):
                hand_label = handedness[idx][0].category_name
                confidence = handedness[idx][0].score
                landmarks = results.hand_landmarks[idx]

                parsed["all_landmarks"].append(
                    HandLandmarks(
                        landmarks=landmarks,
                        handedness=hand_label,
                        confidence=confidence,
                    )
                )

                if hand_label == "Left":
                    parsed["left_hand"] = landmarks
                elif hand_label == "Right":
                    parsed["right_hand"] = landmarks

        return parsed

    def close(self):
        if hasattr(self._detector, "close"):
            self._detector.close()
