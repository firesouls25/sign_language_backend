import os
import sys
import numpy as np
from typing import Optional, Dict, List
import logging
import cv2

logger = logging.getLogger(__name__)

SIGN_LANGUAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sign_language_model")
DATASET_PATH = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "dataset")

if SIGN_LANGUAGE_MODEL_DIR not in sys.path:
    sys.path.insert(0, SIGN_LANGUAGE_MODEL_DIR)


class KeypointExtractor:
    _instance = None
    _detector = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_detector()
        return cls._instance

    def _initialize_detector(self):
        try:
            from models.mediapipe import MediapipeHandDetector

            model_path = os.path.join(
                SIGN_LANGUAGE_MODEL_DIR, "mediapipe/models/hand_landmarker.task"
            )
            if not os.path.exists(model_path):
                logger.warning(f"MediaPipe model not found at {model_path}")
                self._detector = None
                return

            self._detector = MediapipeHandDetector(
                model_path=model_path, num_hands=2, static_image_mode=True
            )
            logger.info("MediaPipe Hand Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe detector: {e}")
            self._detector = None

    def extract_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        if self._detector is None:
            return {"hands": [], "pose": None, "face": None, "frame_shape": frame.shape}

        try:
            results = self._detector.detect(frame)
            return {
                "left_hand": results.get("left_hand"),
                "right_hand": results.get("right_hand"),
                "all_landmarks": results.get("all_landmarks", []),
                "frame_shape": frame.shape,
            }
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            return {"hands": [], "pose": None, "face": None, "frame_shape": frame.shape}

    def close(self):
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass


class MediapipeResults:
    """Wrapper class to make MediaPipe results compatible with SignRecorder"""

    def __init__(self, left_landmarks, right_landmarks):
        self.left_hand_landmarks = left_landmarks
        self.right_hand_landmarks = right_landmarks


class SignRecognizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.sequence_buffer: List[Dict] = []
        self.sequence_length = 30
        self.frame_count = 0
        self.keypoint_extractor = KeypointExtractor()
        self.recorder = None
        self.reference_signs = None
        self._initialized = False

        self._load_model()

    def _load_model(self):
        try:
            from utils.dataset_lsc70_utils import load_reference_signs_lsc70

            logger.info(f"Loading dataset from: {DATASET_PATH}")

            if not os.path.exists(DATASET_PATH):
                logger.error(f"Dataset path does not exist: {DATASET_PATH}")
                self._initialized = False
                return

            original_dir = os.getcwd()

            if os.path.isdir(SIGN_LANGUAGE_MODEL_DIR):
                os.chdir(SIGN_LANGUAGE_MODEL_DIR)
                logger.info(f"Changed directory to: {os.getcwd()}")

            logger.info("Loading LSC70 reference dataset...")
            self.reference_signs = load_reference_signs_lsc70()

            if self.reference_signs is not None and len(self.reference_signs) > 0:
                from sign_recorder import SignRecorder

                self.recorder = SignRecorder(
                    reference_signs=self.reference_signs,
                    min_frames=10,
                    max_frames=60,
                    pause_threshold_frames=8,
                    confidence_threshold=0.6,
                    cooldown_seconds=0.8,
                )
                self._initialized = True
                logger.info(
                    f"SignRecognizer initialized with {len(self.reference_signs)} reference samples"
                )
                unique_signs = self.reference_signs["name"].unique().tolist()
                logger.info(f"Supported signs: {unique_signs}")
            else:
                logger.warning(
                    "No reference signs loaded - SignRecorder not initialized"
                )
                self._initialized = False

            os.chdir(original_dir)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            self._initialized = False

    def process_frame(self, frame: np.ndarray) -> Dict:
        self.frame_count += 1

        self.sequence_buffer.append({"frame": self.frame_count})
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        result = self._recognize_sign(frame)

        return {
            "keypoints": result.get("keypoints"),
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "sequence_length": len(self.sequence_buffer),
            "frame_count": self.frame_count,
            "phrase": result.get("phrase", ""),
            "is_recording": result.get("is_recording", False),
            "candidate": result.get("candidate", ""),
            "candidate_confidence": result.get("candidate_confidence", 0.0),
        }

    def _recognize_sign(self, frame: np.ndarray) -> Dict:
        if not self._initialized or self.recorder is None:
            if len(self.sequence_buffer) < 5:
                return {"text": "", "confidence": 0.0}
            return {
                "text": "Modelo no cargado",
                "confidence": 0.0,
                "phrase": "",
                "is_recording": False,
                "candidate": "",
                "candidate_confidence": 0.0,
            }

        try:
            # Extract keypoints
            keypoints = self.keypoint_extractor.extract_keypoints(frame)

            # Create results object for SignRecorder
            results = MediapipeResults(
                left_landmarks=keypoints.get("left_hand"),
                right_landmarks=keypoints.get("right_hand"),
            )

            # Process through SignRecorder
            sign_detected, is_recording = self.recorder.process_results(results)

            return {
                "keypoints": keypoints.get("left_hand") or keypoints.get("right_hand"),
                "text": self.recorder.current_sign if sign_detected else "",
                "confidence": self.recorder.current_confidence
                if sign_detected
                else 0.0,
                "phrase": self.recorder.get_phrase(),
                "is_recording": is_recording,
                "candidate": self.recorder.candidate_sign if not sign_detected else "",
                "candidate_confidence": self.recorder.candidate_confidence
                if not sign_detected
                else 0.0,
            }

        except Exception as e:
            logger.error(f"Error in sign recognition: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "phrase": "",
                "is_recording": False,
                "candidate": "",
                "candidate_confidence": 0.0,
            }

    def process_landmarks_data(self, results: MediapipeResults) -> Dict:
        """Procesar landmarks directamente sin necesidad de frame"""
        self.frame_count += 1

        self.sequence_buffer.append({"frame": self.frame_count})
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        return self._recognize_from_landmarks(results)

    def _recognize_from_landmarks(self, results: MediapipeResults) -> Dict:
        """Reconocimiento de señas desde landmarks"""
        if not self._initialized or self.recorder is None:
            if len(self.sequence_buffer) < 5:
                return {"text": "", "confidence": 0.0}
            return {
                "text": "Modelo no cargado",
                "confidence": 0.0,
                "phrase": "",
                "is_recording": False,
                "candidate": "",
                "candidate_confidence": 0.0,
            }

        try:
            sign_detected, is_recording = self.recorder.process_results(results)

            return {
                "keypoints": results.left_hand_landmarks
                or results.right_hand_landmarks,
                "text": self.recorder.current_sign if sign_detected else "",
                "confidence": self.recorder.current_confidence
                if sign_detected
                else 0.0,
                "phrase": self.recorder.get_phrase(),
                "is_recording": is_recording,
                "candidate": self.recorder.candidate_sign if not sign_detected else "",
                "candidate_confidence": self.recorder.candidate_confidence
                if not sign_detected
                else 0.0,
            }

        except Exception as e:
            logger.error(f"Error in landmark recognition: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "phrase": "",
                "is_recording": False,
                "candidate": "",
                "candidate_confidence": 0.0,
            }

    def reset_sequence(self):
        self.sequence_buffer = []
        self.frame_count = 0
        if self.recorder is not None:
            self.recorder.clear_phrase()
            self.recorder._reset_state()

    def close(self):
        if self.keypoint_extractor is not None:
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
