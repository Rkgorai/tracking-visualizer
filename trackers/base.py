from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Track:
    """Standardized track representation."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float = 1.0
    class_id: int = 0
    class_name: str = ""
    age: int = 0
    is_confirmed: bool = False
    feature: Optional[np.ndarray] = None
    trajectory: list = field(default_factory=list)  # List of (cx, cy) history

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


class BaseTracker(ABC):
    """Abstract base class for all trackers."""

    def __init__(self, name: str = "BaseTracker", **kwargs):
        self.name = name
        self.frame_count = 0
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    @abstractmethod
    def update(self, detections: list) -> list[Track]:
        """Process detections and return active tracks."""

    def predict(self) -> list[Track]:
        """Predict next state for all active tracks (used by Kalman-based trackers)."""
        return list(self.tracks.values())

    def reset(self) -> None:
        """Reset tracker state."""
        self.frame_count = 0
        self.tracks.clear()
        self._next_id = 0

    def _next_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def iou_batch(bboxes: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU between two sets of boxes."""
        lt = np.maximum(bboxes[:, None, :2], candidates[None, :, :2])
        rb = np.minimum(bboxes[:, None, 2:], candidates[None, :, 2:])
        wh = np.maximum(0, rb - lt)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area_b = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-7)

    def __repr__(self) -> str:
        return f"{self.name}(tracks={len(self.tracks)})"