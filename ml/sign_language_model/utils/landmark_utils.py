import cv2
import os
import numpy as np
import pickle as pkl
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.mediapipe_utils import (
    mediapipe_detection,
    create_hand_model,
    create_pose_model,
)


def landmark_to_array(mp_landmark_list):
    keypoints = []
    for landmark in mp_landmark_list:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    pose = (
        landmark_to_array(results.pose_landmarks).reshape(99).tolist()
        if results.pose_landmarks
        else np.zeros(99).tolist()
    )

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand


def save_landmarks_from_video(video_name):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]

    cap = cv2.VideoCapture(
        os.path.join("data", "videos", sign_name, video_name + ".mp4")
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    hands_model = create_hand_model(
        model_path="mediapipe/models/hand_landmarker.task",
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    pose_model = create_pose_model(
        running_mode=vision.RunningMode.VIDEO,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            timestamp_ms = int(frame_idx * 1000 / fps)
            image, results = mediapipe_detection(
                frame, hands_model, pose_model, timestamp_ms
            )

            pose_landmarks, left_hand, right_hand = extract_landmarks(results)
            landmark_list["pose"].append(pose_landmarks)
            landmark_list["left_hand"].append(left_hand)
            landmark_list["right_hand"].append(right_hand)
            frame_idx += 1
        else:
            break
    cap.release()

    path = os.path.join("data", "dataset", sign_name)
    if not os.path.exists(path):
        os.mkdir(path)

    data_path = os.path.join(path, video_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
    )
    save_array(
        landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_name}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_name}.pickle")
    )


def save_array(arr, path):
    with open(path, "wb") as f:
        pkl.dump(arr, f)


def load_array(path):
    with open(path, "rb") as f:
        return np.array(pkl.load(f))
