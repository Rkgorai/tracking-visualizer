#!/usr/bin/env python3
"""
Tracker Testing Script

Run all trackers on a video with a given model and compare results.
Usage: python test_trackers.py --video path/to/video.mp4 --model yolov8n.pt
       python test_trackers.py --video path/to/video.mp4 --model path/to/custom.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detectors import YOLOv8Detector
from trackers import (
    SortTracker, ByteTrack, OCSortTracker, DeepSORT,
    StrongSORT, MOTRTracker
)
from utils.video import VideoLoader
from utils.visualization import draw_tracks, draw_fps


# Configure logging
def setup_logger(name: str = "tracker_test") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_all_trackers():
    """Return dict of all available trackers with lower thresholds for better detection."""
    return {
        "SORT": lambda: SortTracker(max_age=30, min_hits=1, iou_threshold=0.3),
        "ByteTrack": lambda: ByteTrack(track_thresh=0.3, lost_track_buffer=30, track_iou_threshold=0.3),
        "OC-SORT": lambda: OCSortTracker(det_thresh=0.3, max_age=30, min_hits=1, iou_threshold=0.3, delta_t=3),
        "DeepSORT": lambda: DeepSORT(max_age=30, min_hits=1, iou_threshold=0.3, max_cosine_distance=0.2),
        "StrongSORT": lambda: StrongSORT(max_age=30, min_hits=1, iou_threshold=0.3, ema_alpha=0.9),
        "MOTR": lambda: MOTRTracker(max_age=30, min_hits=1),
    }


def run_tracker(
    video_path: str,
    tracker_name: str,
    detector: YOLOv8Detector,
    output_dir: Path,
    logger: logging.Logger,
    conf_threshold: float = 0.5,
    frame_skip: int = 1,
    max_width: int = 1920,
    filter_classes: list = None,
    total_frames: int = None,
    pbar: tqdm = None,
    start_frame: int = 0
) -> dict:
    """Run a single tracker on the video and return results."""
    logger.info(f"Starting tracker: {tracker_name}")

    # Initialize tracker
    trackers_dict = get_all_trackers()
    if tracker_name not in trackers_dict:
        logger.error(f"Unknown tracker '{tracker_name}'")
        return None

    tracker = trackers_dict[tracker_name]()

    results = {
        "tracker": tracker_name,
        "total_frames": 0,
        "total_tracks": 0,
        "total_detections": 0,
        "fps": 0.0,
        "success": False,
        "error": None
    }

    try:
        with VideoLoader(video_path) as loader:
            if total_frames is None:
                total_frames = loader.total_frames
            fps = loader.fps
            width, height = loader.resolution
            results["total_frames"] = total_frames

            # Create output video
            output_video = output_dir / f"{tracker_name}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

            start_time = time.time()
            all_track_ids = set()
            all_detections = 0

            # Create progress bar for this tracker if not provided
            if pbar is None:
                pbar = tqdm(total=total_frames, desc=tracker_name, unit="frame", leave=False)

            frame_count = 0
            last_logged_frame = -100

            for frame_idx, frame in enumerate(loader.frames()):
                if isinstance(frame, tuple):
                    frame = frame[1] if len(frame) > 1 else frame[0]
                if not isinstance(frame, np.ndarray):
                    pbar.update(1)
                    continue

                # Skip to start frame
                if frame_idx < start_frame:
                    pbar.update(1)
                    continue

                # Skip frames
                if frame_idx % frame_skip != 0:
                    # Still update tracker to maintain continuity
                    tracker.update([])
                    pbar.update(1)
                    continue

                # Resize for detection (skip if max_width is 0)
                h, w = frame.shape[:2]
                if max_width > 0 and w > max_width:
                    scale = max_width / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                else:
                    frame_resized = frame
                    scale = 1.0

                # Detect
                detections = detector.detect(frame_resized)

                # Scale bboxes back
                for det in detections:
                    det.bbox = det.bbox * np.array([scale, scale, scale, scale])

                # Filter by class
                if filter_classes:
                    detections = [d for d in detections if d.class_id in filter_classes]

                all_detections += len(detections)

                # Track
                try:
                    tracks = tracker.update(detections)
                except Exception as e:
                    logger.error(f"Error at frame {frame_idx}: {e}")
                    raise

                # Log detection count every 50 frames
                if frame_idx - last_logged_frame >= 50:
                    logger.info(f"  Frame {frame_idx}: {len(detections)} detections → {len(tracks)} tracks")
                    last_logged_frame = frame_idx

                # Collect track IDs
                for t in tracks:
                    all_track_ids.add(t.track_id)

                # Draw and write
                frame_vis = frame.copy()
                frame_vis = draw_tracks(frame_vis, tracks)
                frame_vis = draw_fps(frame_vis, 0)
                out.write(frame_vis)

                pbar.update(1)
                frame_count += 1

            pbar.close()
            out.release()

            elapsed = time.time() - start_time
            results["fps"] = total_frames / elapsed if elapsed > 0 else 0
            results["total_tracks"] = len(all_track_ids)
            results["total_detections"] = all_detections
            results["success"] = True
            results["output_video"] = str(output_video)
            results["elapsed_time"] = elapsed

            logger.info(f"✓ {tracker_name}: {len(all_track_ids)} tracks, {all_detections} detections, {results['fps']:.1f} FPS ({elapsed:.1f}s)")

    except Exception as e:
        results["error"] = str(e)
        if pbar:
            pbar.close()
        logger.error(f"✗ {tracker_name}: {e}")

    return results


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Test all trackers on a video")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--frame-skip", "-f", type=int, default=1, help="Frame skip (1 = process all)")
    parser.add_argument("--max-width", "-w", type=int, default=1920, help="Max frame width (0 to disable resize)")
    parser.add_argument("--classes", nargs="+", type=int, help="Class IDs to track (e.g., 0 2)")
    parser.add_argument("--start-frame", "-s", type=int, default=0, help="Start frame number")
    parser.add_argument("--output", "-o", default="output/test_results", help="Output directory")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video).exists():
        logger.error(f"Video not found: {args.video}")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("TRACKER TESTING SUITE")
    logger.info("="*60)
    logger.info(f"Video:     {args.video}")
    logger.info(f"Model:     {args.model}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Confidence: {args.conf}")
    logger.info(f"Classes:   {args.classes or 'All'}")
    logger.info(f"Start:     frame {args.start_frame}")
    logger.info("="*60)

    # Initialize detector
    logger.info("Loading detector...")
    detector = YOLOv8Detector(model_path=args.model, conf_threshold=args.conf)
    detector.warmup()

    # Get class names
    class_names = detector.get_class_names()
    logger.info(f"Model classes: {list(class_names.values())}")

    # Get total frames
    with VideoLoader(args.video) as loader:
        total_frames = loader.total_frames

    logger.info(f"Total frames: {total_frames}")
    logger.info("="*60)

    # Run all trackers with overall progress bar
    trackers = get_all_trackers()
    results = []

    # Calculate estimated total time (will be updated as trackers finish)
    overall_pbar = tqdm(total=len(trackers), desc="Trackers", unit="tracker")

    for tracker_name in trackers.keys():
        start_time = time.time()

        result = run_tracker(
            args.video,
            tracker_name,
            detector,
            output_dir,
            logger,
            conf_threshold=args.conf,
            frame_skip=args.frame_skip,
            max_width=args.max_width,
            filter_classes=args.classes,
            total_frames=total_frames,
            start_frame=args.start_frame
        )

        if result:
            results.append(result)

        elapsed = time.time() - start_time
        overall_pbar.set_postfix({
            "last": tracker_name,
            "time": f"{elapsed:.1f}s"
        })
        overall_pbar.update(1)

    overall_pbar.close()

    # Summary
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Tracker':<15} {'Status':<10} {'Tracks':<10} {'FPS':<10} {'Time':<10}")
    logger.info("-"*60)

    for r in results:
        status = "✓ OK" if r["success"] else "✗ FAIL"
        elapsed_str = f"{r.get('elapsed_time', 0):.1f}s" if r["success"] else "-"
        logger.info(f"{r['tracker']:<15} {status:<10} {r['total_tracks']:<10} {r['fps']:<10.1f} {elapsed_str:<10}")

    # Save summary to file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("Tracker Testing Results\n")
        f.write("="*60 + "\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Confidence: {args.conf}\n")
        f.write(f"Classes: {args.classes or 'All'}\n\n")
        f.write(f"{'Tracker':<15} {'Status':<15} {'Tracks':<10} {'FPS':<10} {'Time':<10}\n")
        f.write("-"*60 + "\n")
        for r in results:
            status = "SUCCESS" if r["success"] else f"FAILED: {r['error']}"
            elapsed_str = f"{r.get('elapsed_time', 0):.1f}s" if r["success"] else "-"
            f.write(f"{r['tracker']:<15} {status:<15} {r['total_tracks']:<10} {r['fps']:<10.1f} {elapsed_str:<10}\n")

    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()