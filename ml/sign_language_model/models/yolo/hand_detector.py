from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class YOLOHandResult:
    bbox: List[float]
    confidence: float
    landmarks: Optional[List[List[float]]] = None


class YOLOHandDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_name)

    def detect(self, image) -> List[YOLOHandResult]:
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    detections.append(
                        YOLOHandResult(
                            bbox=box.tolist(),
                            confidence=float(conf),
                        )
                    )

        return detections

    def detect_and_draw(self, image) -> tuple:
        detections = self.detect(image)
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            label = f"hand: {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return detections, annotated
