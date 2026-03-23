import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.mediapipe import MediapipeHolisticDetector


def create_hand_model(
    model_path="mediapipe/models/hand_landmarker.task",
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_pose_model(
    running_mode=vision.RunningMode.VIDEO,
):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="mediapipe/models/pose_landmarker.task"),
        running_mode=running_mode,
    )
    return vision.PoseLandmarker.create_from_options(options)


class MediapipeHolisticResults:
    def __init__(self, hand_results=None, pose_results=None):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.pose_landmarks = None

        if hand_results is not None and hasattr(hand_results, "hand_landmarks"):
            if len(hand_results.hand_landmarks) > 0:
                handedness = hand_results.handedness
                for idx in range(len(hand_results.hand_landmarks)):
                    hand_label = handedness[idx][0].category_name
                    if hand_label == "Left":
                        self.left_hand_landmarks = hand_results.hand_landmarks[idx]
                    elif hand_label == "Right":
                        self.right_hand_landmarks = hand_results.hand_landmarks[idx]

        if pose_results is not None and pose_results.pose_landmarks:
            self.pose_landmarks = pose_results.pose_landmarks[0]


def mediapipe_detection(image, hands_model, pose_model=None, timestamp_ms=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    hand_results = hands_model.detect_for_video(mp_image, timestamp_ms)

    pose_results = None
    if pose_model is not None:
        pose_results = pose_model.detect_for_video(mp_image, timestamp_ms)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = MediapipeHolisticResults(hand_results, pose_results)
    return image, results


def draw_landmarks(image, results):
    from mediapipe.tasks.python.vision import HandLandmarksConnections, drawing_utils

    if results.left_hand_landmarks:
        image = drawing_utils.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=4
            ),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )

    if results.right_hand_landmarks:
        image = drawing_utils.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=4
            ),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )
    return image
