import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections, drawing_utils


WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)
GREEN_COLOR = (35, 220, 80)

WINDOW_NAME = "LSC - Sign Language Recognition"


class WebcamManager(object):
    def __init__(self):
        self.sign_detected = ""
        self.phrase = ""
        self.is_recording = False
        self.confidence = 0.0
        self.candidate = ""
        self.candidate_confidence = 0.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    def update(
        self,
        frame: np.ndarray,
        results,
        sign_detected: str = "",
        is_recording: bool = False,
        phrase: str = "",
        confidence: float = 0.0,
        candidate: str = "",
        candidate_confidence: float = 0.0,
    ):
        self.sign_detected = sign_detected
        self.phrase = phrase
        self.is_recording = is_recording
        self.confidence = confidence
        self.candidate = candidate
        self.candidate_confidence = candidate_confidence

        frame = frame.copy()
        frame = self.draw_landmarks(frame, results)
        frame = cv2.flip(frame, 1)
        frame = self.draw_ui(frame)

        cv2.imshow(WINDOW_NAME, frame)

    def draw_ui(self, frame):
        window_w = len(frame[0])
        window_h = len(frame)

        bottom_bar_h = 120
        bottom_y = window_h - bottom_bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bottom_y), (window_w, window_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (0, bottom_y), (window_w, window_h), (60, 60, 60), 2)

        cv2.putText(
            frame,
            "vista: mediapipe",
            (window_w - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            GREEN_COLOR,
            2,
        )

        if self.sign_detected:
            conf_text = f"{self.confidence:.0%}" if self.confidence > 0 else ""
            text = f"senal: {self.sign_detected}"
            if conf_text:
                text += f" ({conf_text})"
            cv2.putText(
                frame,
                text,
                (30, window_h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                GREEN_COLOR,
                3,
            )
        elif self.candidate:
            conf_text = f"{self.candidate_confidence:.0%}" if self.candidate_confidence > 0 else ""
            text = f"senal: {self.candidate}?"
            if conf_text:
                text += f" ({conf_text})"
            cv2.putText(
                frame,
                text,
                (30, window_h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (150, 200, 255),
                3,
            )
        elif self.is_recording:
            cv2.putText(
                frame,
                "reconociendo...",
                (30, window_h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (200, 200, 100),
                2,
            )
        else:
            cv2.putText(
                frame,
                "esperando...",
                (30, window_h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                WHITE_COLOR,
                2,
            )

        if self.phrase:
            phrase_display = f"frase: {self.phrase}"
            cv2.putText(
                frame,
                phrase_display,
                (30, window_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 255),
                2,
            )

        color = WHITE_COLOR
        if self.is_recording:
            color = RED_COLOR
        cv2.circle(frame, (window_w - 40, 70), 15, color, -1)
        if color == RED_COLOR:
            cv2.circle(frame, (window_w - 40, 70), 15, (255, 255, 255), 2)

        cv2.putText(
            frame,
            "q=salir  c=borrar",
            (window_w - 250, window_h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
        )

        return frame

    @staticmethod
    def draw_landmarks(image, results):
        if results.left_hand_landmarks:
            drawing_utils.draw_landmarks(
                image,
                landmark_list=results.left_hand_landmarks,
                connections=HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(
                    color=(232, 254, 255), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(255, 249, 161), thickness=2, circle_radius=2
                ),
            )

        if results.right_hand_landmarks:
            drawing_utils.draw_landmarks(
                image,
                landmark_list=results.right_hand_landmarks,
                connections=HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(
                    color=(232, 254, 255), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(255, 249, 161), thickness=2, circle_radius=2
                ),
            )
        return image
