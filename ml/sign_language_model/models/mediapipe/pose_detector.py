from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class PoseLandmarks:
    landmarks: List
    confidence: float


class MediapipePoseDetector:
    def __init__(
        self,
        model_path: str = "mediapipe/models/pose_landmarker.task",
        running_mode: str = "VIDEO",
        static_image_mode: bool = False,
    ):
        self.model_path = model_path
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
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
        )
        return vision.PoseLandmarker.create_from_options(options)

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
            "pose": None,
        }

        if results is None or not results.pose_landmarks:
            return parsed

        parsed["pose"] = results.pose_landmarks[0]
        return parsed

    def close(self):
        if hasattr(self._detector, "close"):
            self._detector.close()
