import numpy as np
from dataclasses import dataclass, field

from typing import Optional


@dataclass
class Metrics:
    """Tracking metrics."""

    mota: float = 0.0  # Multiple Object Tracking Accuracy
    idf1: float = 0.0  # ID F1 Score
    fps: float = 0.0   # Frames per second
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    num_misses: int = 0      # False negatives
    num_false_positives: int = 0
    num_id_switches: int = 0
    num_fragmentations: int = 0
    total_objects: int = 0
    total_predictions: int = 0
    track_ids: set = field(default_factory=set)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_mota(
    ground_truth: list,
    predictions: list,
    iou_threshold: float = 0.5
) -> Metrics:
    """
    Compute MOTA and related metrics.

    Args:
        ground_truth: List of (frame_id, track_id, bbox) tuples
        predictions: List of (frame_id, track_id, bbox) tuples
        iou_threshold: Minimum IoU for a match
    """
    metrics = Metrics()

    if not ground_truth:
        return metrics

    # Group by frame
    gt_by_frame = {}
    for frame_id, track_id, bbox in ground_truth:
        if frame_id not in gt_by_frame:
            gt_by_frame[frame_id] = {}
        gt_by_frame[frame_id][track_id] = bbox

    pred_by_frame = {}
    for frame_id, track_id, bbox in predictions:
        if frame_id not in pred_by_frame:
            pred_by_frame[frame_id] = {}
        pred_by_frame[frame_id][track_id] = bbox

    # Track ID mapping for ID switches
    id_mapping = {}  # frame -> {pred_id: gt_id}
    matched_gt = {}  # gt_id -> best matched pred_id over time

    num_misses = 0
    num_fp = 0
    num_ids = 0

    total_gt = 0

    for frame_id in sorted(gt_by_frame.keys()):
        gt_tracks = gt_by_frame.get(frame_id, {})
        pred_tracks = pred_by_frame.get(frame_id, {})

        total_gt += len(gt_tracks)

        if not pred_tracks:
            num_misses += len(gt_tracks)
            continue

        # Match GT to predictions using IoU
        matched = {}
        for gt_id, gt_bbox in gt_tracks.items():
            best_iou = 0
            best_pred_id = None
            for pred_id, pred_bbox in pred_tracks.items():
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_id = pred_id

            if best_pred_id is not None:
                matched[gt_id] = best_pred_id
                # Check for ID switch
                if frame_id > 0 and gt_id in id_mapping.get(frame_id - 1, {}):
                    prev_pred = id_mapping[frame_id - 1].get(gt_id)
                    if prev_pred != best_pred_id:
                        num_ids += 1

        # Count false positives
        matched_pred_ids = set(matched.values())
        for pred_id in pred_tracks:
            if pred_id not in matched_pred_ids:
                num_fp += 1

        # Count misses
        num_misses += len(gt_tracks) - len(matched)

        # Store mapping for next frame
        id_mapping[frame_id] = matched

    # Compute MOTA
    mota = 1 - (num_misses + num_fp + num_ids) / max(total_gt, 1)
    metrics.mota = max(0, mota)
    metrics.num_misses = num_misses
    metrics.num_false_positives = num_fp
    metrics.num_id_switches = num_ids
    metrics.total_objects = total_gt
    metrics.total_predictions = sum(len(p) for p in pred_by_frame.values())

    # Precision and Recall
    if metrics.total_predictions > 0:
        metrics.precision = (total_gt - num_misses) / metrics.total_predictions
    if total_gt > 0:
        metrics.recall = (total_gt - num_misses) / total_gt

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    # Collect all unique track IDs
    all_gt_ids = set(gt_id for _, gt_id, _ in ground_truth)
    all_pred_ids = set(pred_id for _, pred_id, _ in predictions)
    metrics.track_ids = all_gt_ids | all_pred_ids

    return metrics


def compute_idf1(ground_truth: list, predictions: list) -> float:
    """
    Compute ID F1 Score.

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """
    if not ground_truth or not predictions:
        return 0.0

    # ID True Positives: matches with same ID
    # ID False Positives: predictions with wrong ID
    # ID False Negatives: ground truth with wrong ID

    gt_by_frame = {}
    for frame_id, track_id, bbox in ground_truth:
        if frame_id not in gt_by_frame:
            gt_by_frame[frame_id] = {}
        gt_by_frame[frame_id][track_id] = bbox

    pred_by_frame = {}
    for frame_id, track_id, bbox in predictions:
        if frame_id not in pred_by_frame:
            pred_by_frame[frame_id] = {}
        pred_by_frame[frame_id][track_id] = bbox

    idtp = 0
    idfp = 0
    idfn = 0

    for frame_id in sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys())):
        gt_tracks = gt_by_frame.get(frame_id, {})
        pred_tracks = pred_by_frame.get(frame_id, {})

        gt_ids = set(gt_tracks.keys())
        pred_ids = set(pred_tracks.keys())

        common_ids = gt_ids & pred_ids

        for gt_id in gt_ids:
            if gt_id in common_ids:
                idtp += 1
            else:
                idfn += 1

        for pred_id in pred_ids:
            if pred_id not in common_ids:
                idfp += 1

    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
    return idf1


def compute_fps(num_frames: int, elapsed_time: float) -> float:
    """Compute frames per second."""
    return num_frames / elapsed_time if elapsed_time > 0 else 0.0