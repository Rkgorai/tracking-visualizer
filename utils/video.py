from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


class VideoLoader:
    """Load and iterate over video frames."""

    def __init__(self, source: str):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._total: int = 0

    def open(self) -> "VideoLoader":
        src = self.source
        if isinstance(src, str) and Path(src).exists():
            src = str(Path(src).absolute())
        self.cap = cv2.VideoCapture(src)
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def close(self) -> None:
        if self.cap:
            self.cap.release()

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_idx, frame) pairs."""
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def total_frames(self) -> int:
        return self._total

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


class VideoWriter:
    """Write annotated frames to video file."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self.out.write(frame)

    def release(self) -> None:
        self.out.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()