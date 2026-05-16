import numpy as np
import cv2

from detectors.base import BaseDetector, Detection


class YOLOv8Detector(BaseDetector):
    """YOLOv8 object detector using Ultralytics."""

    # COCO class names
    COCO_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
        35: "baseball glove", 36: "surfboard", 37: "tennis racket", 38: "bottle",
        39: "wine glass", 40: "cup", 41: "fork", 42: "knife", 43: "spoon",
        44: "bowl", 45: "banana", 46: "apple", 47: "sandwich", 48: "orange",
        49: "broccoli", 50: "carrot", 51: "hot dog", 52: "pizza", 53: "donut",
        54: "cake", 55: "chair", 56: "couch", 57: "potted plant", 58: "bed",
        59: "dining table", 60: "toilet", 61: "tv", 62: "laptop", 63: "mouse",
        64: "remote", 65: "keyboard", 66: "cell phone", 67: "microwave", 68: "oven",
        69: "toaster", 70: "sink", 71: "refrigerator", 72: "book", 73: "clock",
        74: "vase", 75: "scissors", 76: "teddy bear", 77: "hair drier", 78: "toothbrush"
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5,
                 device: str = "cpu", classes: list = None):
        super().__init__(model_path, conf_threshold, device, classes or self.COCO_CLASSES)
        self._model = None

    def load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load_model()

        # Ensure image is valid numpy array with correct dtype
        if image is None or not isinstance(image, np.ndarray):
            return []

        # Ensure image is uint8 and has 3 channels
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0])
                class_id = int(box.cls[0])

                class_name = self.classes[class_id] if self.classes else str(class_id)

                detection = Detection(
                    bbox=xyxy,
                    score=score,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

        return detections

    def warmup(self) -> None:
        if self._model is None:
            self.load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)