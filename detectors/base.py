from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """Standardized detection output."""

    bbox: np.ndarray  # [x1, y1, x2, y2] or [x, y, w, h]
    score: float
    class_id: int
    class_name: str = ""
    feature: Optional[np.ndarray] = None  # Appearance embedding

    @property
    def xyxy(self) -> np.ndarray:
        return self.bbox

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class BaseDetector(ABC):
    """Abstract base class for all object detectors."""

    def __init__(self, model_path: str = "", conf_threshold: float = 0.5,
                 device: str = "cpu", classes: Optional[list] = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on an image and return standardized detections."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass to warm up the model."""

    def set_classes(self, classes: list) -> None:
        self.classes = classes

    def set_conf_threshold(self, threshold: float) -> None:
        self.conf_threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(conf={self.conf_threshold}, device={self.device})"