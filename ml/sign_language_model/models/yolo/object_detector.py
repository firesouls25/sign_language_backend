from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class YOLODetection:
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float


class YOLOObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
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

        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict:
        try:
            return self.model.names
        except Exception:
            return {}

    def detect(self, image) -> List[YOLODetection]:
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
                    cls_int = int(cls)
                    class_name = self.class_names.get(cls_int, f"class_{cls_int}")

                    detections.append(
                        YOLODetection(
                            class_id=cls_int,
                            class_name=class_name,
                            bbox=box.tolist(),
                            confidence=float(conf),
                        )
                    )

        return detections

    def detect_filtered(
        self, image, class_names: Optional[List[str]] = None
    ) -> List[YOLODetection]:
        all_detections = self.detect(image)

        if class_names:
            return [d for d in all_detections if d.class_name in class_names]
        return all_detections

    def detect_and_draw(
        self,
        image,
        color_map: Optional[dict] = None,
    ) -> tuple:
        detections = self.detect(image)
        annotated = image.copy()

        if color_map is None:
            np.random.seed(42)
            color_map = {}
            for cls_id, cls_name in self.class_names.items():
                color_map[cls_name] = tuple(map(int, np.random.randint(0, 255, 3).tolist()))

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = color_map.get(det.class_name, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return detections, annotated
