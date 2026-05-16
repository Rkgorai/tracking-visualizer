# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Run the Streamlit app
streamlit run app.py
```

## Project Overview

**Tracking Visualizer** is a Streamlit-based multi-object tracking (MOT) comparison tool. It runs various tracking algorithms (SORT, ByteTrack, OC-SORT, DeepSORT, StrongSORT, MOTR) with YOLOv8 detectors on user-uploaded videos and displays results with metrics.

## Architecture

### Core Components

- **`app.py`** — Main Streamlit application. Entry point for the UI. Contains `TRACKER_OPTIONS` and `DETECTOR_OPTIONS` dictionaries that register all available trackers/detectors. The `process_video()` function handles the detection → tracking → visualization pipeline.

- **`trackers/`** — Tracking algorithms organized by family:
  - `base.py` — `BaseTracker` abstract class and `Track` dataclass. All trackers inherit from `BaseTracker` and implement `update(detections: list) -> list[Track]`. Provides `iou()` and `iou_batch()` static methods.
  - `traditional/` — Kalman Filter and SORT implementation
  - `botsort_family/` — ByteTrack, OC-SORT, BoT-SORT
  - `deepsort_family/` — DeepSORT, StrongSORT
  - `transformer_based/` — MOTR

- **`detectors/`** — Object detection:
  - `base.py` — `BaseDetector` abstract class and `Detection` dataclass
  - `yolov8.py` — YOLOv8 implementation using Ultralytics. Supports yolov8n.pt, yolov8s.pt, yolov8m.pt models

- **`utils/`** — Utility modules:
  - `video.py` — `VideoLoader`, `VideoWriter` for frame I/O
  - `metrics.py` — MOTA, IDF1, FPS calculation
  - `visualization.py` — `draw_tracks()`, `draw_fps()` for annotating frames
  - `export.py` — MOT format, CSV, JSON export

### Data Flow

1. Video uploaded → `VideoLoader` reads frames
2. Each frame → `YOLOv8Detector.detect()` returns `list[Detection]`
3. Detections → `tracker.update()` returns `list[Track]`
4. Tracks → `draw_tracks()` annotates frame
5. Annotated frames → `VideoWriter` outputs MP4
6. Track data → `export.py` writes MOT/CSV/JSON

### Key Classes

- **`Track`** — Represents a tracked object with `track_id`, `bbox`, `score`, `class_id`, `class_name`, `trajectory`, `feature`
- **`Detection`** — Represents a detection with `bbox`, `score`, `class_id`, `class_name`, `feature`

### Adding a New Tracker

```python
from trackers.base import BaseTracker, Track

class MyTracker(BaseTracker):
    def __init__(self, **kwargs):
        super().__init__(name="MyTracker", **kwargs)

    def update(self, detections: list) -> list[Track]:
        # Implement tracking logic
        pass
```

Then register in `app.py` by adding to `TRACKER_OPTIONS`.

## Output Formats

- **MOT format** — `<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1>`
- **CSV** — `frame,track_id,x1,y1,x2,y2,score,class_id,class_name`
- **JSON** — Array of frames with track details and trajectories

## Dependencies

Key packages: streamlit, opencv-python, torch, ultralytics, numpy, scipy, filterpy, lap, motmetrics

## Notes

- YOLOv8 models are downloaded automatically on first run (yolov8n.pt, yolov8s.pt, yolov8m.pt in project root)
- Frame scaling is applied before detection, then bounding boxes are scaled back to original resolution for tracking
- COCO 80-class taxonomy is used for object classes (person=0, car=2, etc.)