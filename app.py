import streamlit as st
import numpy as np
import cv2
import time
from pathlib import Path
from typing import Optional

from detectors import YOLOv8Detector, Detection
from trackers import (
    SortTracker, ByteTrack, OCSortTracker, DeepSORT,
    StrongSORT, MOTRTracker, BaseTracker
)
from utils.video import VideoLoader, VideoWriter
from utils.visualization import draw_tracks, draw_fps
from utils.metrics import compute_mota, compute_fps
from utils.export import export_mot_format, export_csv, export_json


TRACKER_OPTIONS = {
    "SORT": SortTracker,
    "ByteTrack": ByteTrack,
    "OC-SORT": OCSortTracker,
    "DeepSORT": DeepSORT,
    "StrongSORT": StrongSORT,
    "MOTR": MOTRTracker,
}

DETECTOR_OPTIONS = {
    "YOLOv8-nano": "yolov8n.pt",
    "YOLOv8-small": "yolov8s.pt",
    "YOLOv8-medium": "yolov8m.pt",
}

# Default COCO classes (fallback)
DEFAULT_COCO_CLASSES = {
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


def initialize_tracker(tracker_name: str) -> BaseTracker:
    """Initialize tracker with default parameters."""
    tracker_class = TRACKER_OPTIONS.get(tracker_name, SortTracker)

    if tracker_name == "SORT":
        return tracker_class(max_age=30, min_hits=3, iou_threshold=0.3)
    elif tracker_name == "ByteTrack":
        return tracker_class(track_thresh=0.3, lost_track_buffer=30, track_iou_threshold=0.3)
    elif tracker_name == "OC-SORT":
        return tracker_class(det_thresh=0.3, max_age=30, iou_threshold=0.3, delta_t=3)
    elif tracker_name == "DeepSORT":
        return tracker_class(max_age=30, min_hits=3, iou_threshold=0.3, max_cosine_distance=0.2)
    elif tracker_name == "StrongSORT":
        return tracker_class(max_age=30, min_hits=3, iou_threshold=0.3, ema_alpha=0.9)
    elif tracker_name == "MOTR":
        return tracker_class(max_age=30, min_hits=3)
    else:
        return tracker_class()


def initialize_detector(detector_name: str, conf_threshold: float, custom_classes: dict = None, custom_model_path: str = None) -> YOLOv8Detector:
    """Initialize detector with optional custom model and classes."""
    if custom_model_path:
        model_path = custom_model_path
    else:
        model_path = DETECTOR_OPTIONS.get(detector_name, "yolov8n.pt")

    classes = custom_classes if custom_classes else None
    return YOLOv8Detector(model_path=model_path, conf_threshold=conf_threshold, classes=classes)


def process_video(
    video_path: str,
    tracker_name: str,
    detector_name: str,
    conf_threshold: float,
    progress_bar,
    frame_skip: int = 3,
    max_width: int = 640,
    filter_classes: list = None,
    custom_classes: dict = None,
    custom_model_path: str = None
) -> tuple:
    """Process video and return tracking results. Saves annotated video to disk."""

    # COCO class names mapping (fallback)
    COCO_CLASSES = DEFAULT_COCO_CLASSES

    detector = initialize_detector(detector_name, conf_threshold, custom_classes, custom_model_path)
    detector.warmup()

    tracker = initialize_tracker(tracker_name)

    tracks_per_frame = []
    all_detections = []

    start_time = time.time()

    # Get video properties for writing output
    with VideoLoader(video_path) as loader:
        total_frames = loader.total_frames
        fps = loader.fps
        width, height = loader.resolution
        effective_frames = total_frames // frame_skip

        # Create output video path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_video_path = output_dir / f"tracked_{tracker_name}_{int(time.time())}.mp4"

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # Resize frame if needed
        def resize_frame(frame):
            if isinstance(frame, tuple):
                frame = frame[1] if len(frame) > 1 else frame[0]
            if not isinstance(frame, np.ndarray):
                return frame
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(frame, (new_w, new_h))
            return frame

        for frame_idx, frame in enumerate(loader.frames()):
            # Handle tuple frames
            if isinstance(frame, tuple):
                frame = frame[1] if len(frame) > 1 else frame[0]
            if not isinstance(frame, np.ndarray):
                continue

            # Skip frames - only process every Nth frame
            if frame_idx % frame_skip != 0:
                # Still track with previous detection to maintain continuity
                tracks = tracker.update([])
                tracks_per_frame.append(tracks)
                # Draw tracks on frame and write to output video (use last known tracks)
                frame_vis = frame.copy()
                frame_vis = draw_tracks(frame_vis, tracks)
                out.write(frame_vis)
                continue

            # Resize for faster detection
            frame_resized = resize_frame(frame)

            # Skip if frame is invalid
            if not isinstance(frame_resized, np.ndarray):
                tracks_per_frame.append([])
                continue

            # Detect
            detections = detector.detect(frame_resized)

            # Debug: Print all class IDs for first few frames
            if frame_idx < 3 and detections:
                class_ids = [d.class_id for d in detections]
                class_names = [d.class_name for d in detections]
                print(f"Frame {frame_idx}: Detected class_ids={class_ids}, class_names={class_names}")

            # Filter by class(es) if specified
            if filter_classes is not None:
                detections = [d for d in detections if d.class_id in filter_classes]

            # Debug: Print after filter
            if frame_idx < 3:
                print(f"Frame {frame_idx}: {len(detections)} detections after filter (filter_classes={filter_classes})")

            # Scale bbox back to original size for tracking
            h, w = frame.shape[:2]
            if isinstance(frame_resized, np.ndarray):
                h_r, w_r = frame_resized.shape[:2]
            else:
                h_r, w_r = h, w
            scale_x = w / w_r if w_r > 0 else 1
            scale_y = h / h_r if h_r > 0 else 1

            for det in detections:
                det.bbox = det.bbox * np.array([scale_x, scale_y, scale_x, scale_y])

            # Track
            tracks = tracker.update(detections)

            # Debug: track count
            if frame_idx < 3:
                print(f"Frame {frame_idx}: {len(tracks)} tracks after tracking")

            # Store results
            tracks_per_frame.append(tracks)

            # Store for metrics
            for track in tracks:
                all_detections.append((frame_idx + 1, track.track_id, track.bbox))

            # Draw tracks on frame and write to output video
            frame_vis = frame.copy()
            frame_vis = draw_tracks(frame_vis, tracks)
            out.write(frame_vis)

            # Update progress
            progress_bar.progress((frame_idx // frame_skip + 1) / max(effective_frames, 1))

    # Release video writer
    out.release()

    elapsed_time = time.time() - start_time
    fps_processed = compute_fps(total_frames, elapsed_time)

    return tracks_per_frame, all_detections, fps_processed, total_frames, str(output_video_path)


def main():
    st.set_page_config(page_title="Tracking Visualizer", page_icon="🎯", layout="wide")

    st.title("🎯 Tracking Visualizer")
    st.markdown("Compare traditional and SOTA tracking algorithms on your videos")

    # Sidebar
    st.sidebar.header("Configuration")

    # Video input
    video_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if video_file:
        # Save uploaded video temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        video_path = temp_dir / video_file.name

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Tracker selection
        tracker_name = st.sidebar.selectbox(
            "Select Tracker",
            options=list(TRACKER_OPTIONS.keys()),
            index=0,
            help="Choose a tracking algorithm"
        )

        # Detector selection
        st.sidebar.header("Detector")
        model_type = st.sidebar.radio(
            "Model Type",
            options=["Default YOLOv8", "Custom Model"],
            index=0,
            help="Choose between default YOLOv8 models or upload your own fine-tuned model"
        )

        custom_model_path = None
        custom_classes = None

        if model_type == "Default YOLOv8":
            detector_name = st.sidebar.selectbox(
                "Select Detector",
                options=list(DETECTOR_OPTIONS.keys()),
                index=0,
                help="Choose YOLOv8 model size"
            )
        else:
            # Custom model upload
            uploaded_model = st.sidebar.file_uploader(
                "Upload Custom Model",
                type=["pt", "pth"],
                help="Upload your fine-tuned YOLO model (.pt or .pth)"
            )

            if uploaded_model:
                # Save uploaded model temporarily
                model_dir = Path("temp")
                model_dir.mkdir(exist_ok=True)
                custom_model_path = str(model_dir / uploaded_model.name)

                with open(custom_model_path, "wb") as f:
                    f.write(uploaded_model.read())

                # Auto-detect classes from model
                temp_detector = YOLOv8Detector(model_path=custom_model_path, conf_threshold=0.5)
                custom_classes = temp_detector.get_class_names()

                st.sidebar.success(f"Loaded: {uploaded_model.name} ({len(custom_classes)} classes)")

                detector_name = "Custom"  # marker for custom model
            else:
                detector_name = "YOLOv8-nano"  # default fallback
                custom_classes = DEFAULT_COCO_CLASSES
                st.sidebar.info("Upload a model to use custom classes")

        # Parameters
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

        # Class filtering - what to track
        st.sidebar.header("Object Classes")

        # Determine available classes based on model type
        if model_type == "Default YOLOv8":
            available_classes = DEFAULT_COCO_CLASSES
        else:
            available_classes = custom_classes if custom_classes else DEFAULT_COCO_CLASSES

        # Build class options for selection (class_name -> class_id)
        class_list = list(available_classes.items())

        selected_classes = st.sidebar.multiselect(
            "Track Only (select classes)",
            options=[name for _, name in class_list],
            default=[],
            help="Select which object classes to track (empty = all objects)"
        )

        # Convert selected class names to IDs
        selected_class_ids = [cid for cid, name in class_list if name in selected_classes] if selected_classes else None

        # Performance options
        st.sidebar.header("Performance")
        frame_skip = st.sidebar.selectbox(
            "Frame Skip",
            options=[1, 2, 3, 5, 10],
            index=2,
            help="Process every Nth frame (higher = faster but less accurate)"
        )

        max_width = st.sidebar.selectbox(
            "Max Frame Width",
            options=[320, 480, 640, 960, 1280, 1920, 2560],
            index=4,
            help="Resize frames to this max width for faster detection (larger = more accurate)"
        )

        # Process button
        if st.sidebar.button("Run Tracking", type="primary"):
            filter_info = f" tracking {selected_classes}" if selected_classes else ""
            with st.spinner(f"Processing video{filter_info}... This may take a while."):
                progress_bar = st.progress(0)

                result = process_video(
                    str(video_path),
                    tracker_name,
                    detector_name,
                    conf_threshold,
                    progress_bar,
                    frame_skip=frame_skip,
                    max_width=max_width,
                    filter_classes=selected_class_ids,
                    custom_classes=custom_classes if model_type == "Custom Model" else None,
                    custom_model_path=custom_model_path
                )

                # Handle return values
                tracks_per_frame, all_detections, fps, total_frames, output_video = result

            st.success(f"Processing complete! FPS: {fps:.1f}")

            # Display results
            st.header("Results")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Frames", total_frames)
            col2.metric("Total Tracks", len(set(d[1] for d in all_detections)))
            col3.metric("FPS", f"{fps:.1f}")
            col4.metric("Total Detections", len(all_detections))

            # Check if tracking worked
            if len(all_detections) == 0:
                st.warning(f"No objects detected! Try lowering the confidence threshold (current: {conf_threshold})")
            elif len(set(d[1] for d in all_detections)) == 0:
                st.warning("No tracks created. Check if tracker is working properly.")

            # Play the tracked video
            st.subheader("Tracked Video")
            if output_video and Path(output_video).exists():
                st.video(output_video)

                # Download button
                with open(output_video, "rb") as f:
                    st.download_button(
                        label="Download Tracked Video",
                        data=f,
                        file_name=Path(output_video).name,
                        mime="video/mp4"
                    )
            else:
                st.error("Output video not found")

            # Frame slider for inspection
            st.subheader("Frame Inspector")

            # Frame slider for inspection
            frame_idx = st.slider("Inspect Frame", 0, total_frames - 1, 0)

            # Show frame with tracks
            if frame_idx < len(tracks_per_frame):
                # Get the frame from original video
                with VideoLoader(str(video_path)) as loader:
                    for i, (orig_frame_idx, frame) in enumerate(loader.frames()):
                        if isinstance(orig_frame_idx, tuple):
                            frame = orig_frame_idx[1] if len(orig_frame_idx) > 1 else orig_frame_idx[0]
                        if i == frame_idx:
                            frame = draw_tracks(frame, tracks_per_frame[frame_idx])
                            frame = draw_fps(frame, fps)
                            st.image(frame, caption=f"Frame {frame_idx}", use_column_width=True)
                            break

                # Track details
                tracks = tracks_per_frame[frame_idx]
                if tracks:
                    track_data = []
                    for t in tracks:
                        track_data.append({
                            "ID": t.track_id,
                            "Class": t.class_name,
                            "Score": f"{t.score:.2f}",
                            "BBox": f"[{t.bbox[0]:.0f}, {t.bbox[1]:.0f}, {t.bbox[2]:.0f}, {t.bbox[3]:.0f}]"
                        })
                    st.table(track_data)
                else:
                    st.info("No tracks in this frame")

            # Export options
            st.subheader("Export")

            col1, col2, col3 = st.columns(3)

            # Export to MOT format
            mot_path = "temp/tracking_mot.txt"
            export_mot_format(tracks_per_frame, mot_path)
            with open(mot_path, "rb") as f:
                col1.download_button(
                    "MOT Format",
                    f,
                    "tracking_mot.txt",
                    "text/plain"
                )

            # Export to CSV
            csv_path = "temp/tracking.csv"
            export_csv(tracks_per_frame, csv_path)
            with open(csv_path, "rb") as f:
                col2.download_button(
                    "CSV Format",
                    f,
                    "tracking.csv",
                    "text/csv"
                )

            # Export to JSON
            json_path = "temp/tracking.json"
            export_json(tracks_per_frame, json_path)
            with open(json_path, "rb") as f:
                col3.download_button(
                    "JSON Format",
                    f,
                    "tracking.json",
                    "application/json"
                )

    else:
        # Demo with sample videos
        st.info("Upload a video to start tracking, or use one of the sample videos below.")

        st.markdown("### Available Trackers")
        st.markdown("""
        | Tracker | Type | Description |
        |---------|------|-------------|
        | SORT | Traditional | Kalman Filter + Hungarian algorithm |
        | ByteTrack | Traditional | ByteTrack-style with confidence filtering |
        | OC-SORT | Traditional | Observation-Centric SORT |
        | DeepSORT | SOTA | DeepSORT with ReID features |
        | StrongSORT | SOTA | Enhanced DeepSORT with GSI and MC |
        | MOTR | SOTA | Transformer-based end-to-end tracking |
        """)

        st.markdown("### Available Detectors")
        st.markdown("""
        - **YOLOv8-nano**: Fast, lightweight model
        - **YOLOv8-small**: Balanced speed/accuracy
        - **YOLOv8-medium**: Higher accuracy
        """)


if __name__ == "__main__":
    main()