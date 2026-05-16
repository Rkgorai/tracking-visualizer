# Tracking Visualizer

A comprehensive tracking visualizer for comparing traditional algorithms (Kalman Filter, SORT, ByteTrack, OC-SORT) with state-of-the-art deep learning models (DeepSORT, StrongSORT, BoT-SORT, MOTR).

## Features

- **Multi-tracker comparison**: Compare multiple trackers side-by-side on the same video
- **Real-time visualization**: Interactive frame-by-frame tracking with live metrics
- **Metrics dashboard**: MOTA, IDF1, FPS, precision, recall, and comparison charts
- **Export capabilities**: MOTChallenge format, JSON, CSV, and annotated videos
- **Modular design**: Easy to add new trackers and detectors

## Supported Trackers

| Tracker | Type | Description |
|---------|------|-------------|
| SORT | Traditional | Kalman Filter + Hungarian algorithm for IoU-based matching |
| ByteTrack | Traditional | Multi-object tracking by associating every detection box |
| OC-SORT | Traditional | Observation-Centric SORT with direction-aware matching |
| DeepSORT | SOTA | DeepSORT with ReID appearance features and cosine distance |
| StrongSORT | SOTA | Enhanced DeepSORT with GSI, AFI, and Motion Consistency |
| BoT-SORT | SOTA | BoT-SORT with camera motion compensation and IoU+ReID matching |
| MOTR | SOTA | Transformer-based end-to-end multi-object tracking |

## Supported Detectors

| Detector | Description |
|----------|-------------|
| YOLOv8-nano | Fast, lightweight (3.9M params) |
| YOLOv8-small | Balanced speed/accuracy (11.8M params) |
| YOLOv8-medium | Higher accuracy (25.9M params) |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 models (automatic on first run)
# Models: yolov8n.pt, yolov8s.pt, yolov8m.pt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

### Using the App

1. **Upload a video** (MP4, AVI, or MOV format)
2. **Select a tracker** from the sidebar (SORT, ByteTrack, OC-SORT, DeepSORT, StrongSORT, MOTR)
3. **Select a detector** (YOLOv8-nano, YOLOv8-small, YOLOv8-medium)
4. **Adjust confidence threshold** (default: 0.5)
5. **Click "Run Tracking"** to process the video
6. **Explore results**: View visualization, inspect frames, download results

## Project Structure

```
visual/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── trackers/                  # Tracking algorithms
│   ├── base.py               # BaseTracker, Track classes
│   ├── traditional/          # Kalman Filter, SORT
│   ├── deepsort_family/      # DeepSORT, StrongSORT
│   ├── botsort_family/       # ByteTrack, OC-SORT, BoT-SORT
│   └── transformer_based/    # MOTR
├── detectors/                 # Object detectors
│   ├── base.py               # BaseDetector, Detection classes
│   └── yolov8.py             # YOLOv8 implementation
└── utils/                    # Utility modules
    ├── video.py              # Video loading/writing
    ├── metrics.py            # MOTA, IDF1, FPS calculation
    ├── visualization.py      # Drawing tracks, trajectories
    └── export.py             # MOT, CSV, JSON export
```

## Output Formats

### MOTChallenge Format
```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```
Example: `1,1,100.00,200.00,50.00,80.00,0.95,-1,-1,-1`

### CSV Format
```csv
frame,track_id,x1,y1,x2,y2,score,class_id,class_name
1,1,100.0,200.0,150.0,280.0,0.95,0,person
```

### JSON Format
```json
[
  {
    "frame": 1,
    "tracks": [
      {
        "track_id": 1,
        "bbox": [100.0, 200.0, 150.0, 280.0],
        "score": 0.95,
        "class_id": 0,
        "class_name": "person",
        "trajectory": [[125, 240], ...]
      }
    ]
  }
]
```

## Metrics

- **MOTA** (Multiple Object Tracking Accuracy): Accounts for false positives, false negatives, and ID switches
- **IDF1**: Identity F1 score - ratio of correctly identified objects
- **FPS**: Frames per second - processing speed
- **Precision/Recall**: Detection-level metrics

## Adding New Trackers

```python
from trackers.base import BaseTracker, Track

class MyTracker(BaseTracker):
    def __init__(self, **kwargs):
        super().__init__(name="MyTracker", **kwargs)

    def update(self, detections: list) -> list[Track]:
        # Implement your tracking logic
        pass

# Register in app.py
TRACKER_OPTIONS["MyTracker"] = MyTracker
```

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Streamlit
- Ultralytics (YOLOv8)
- NumPy, SciPy, Pandas

## License

MIT License

## References

- SORT: Simple Online and Realtime Tracking
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box
- OC-SORT: Observation-Centric SORT
- DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric
- StrongSORT: StrongSORT with GSI and Motion Consistency
- BoT-SORT: Robust Associations by Re-Identification in Multi-Object Tracking
- MOTR: End-to-End Multi-Object Tracking with Transformer