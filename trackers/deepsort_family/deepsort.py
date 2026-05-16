import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track


class DeepSORT(BaseTracker):
    """DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        n_init: int = 3,
        **kwargs
    ):
        super().__init__(name="DeepSORT", **kwargs)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.n_init = n_init

        self.feature_cache: dict[int, list] = {}  # Feature history for cosine matching

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        # Update age of existing tracks and feature cache
        for track in self.tracks.values():
            track.age += 1
            if track.track_id not in self.feature_cache:
                self.feature_cache[track.track_id] = []
            if track.feature is not None:
                self.feature_cache[track.track_id].append(track.feature)
                self.feature_cache[track.track_id] = self.feature_cache[track.track_id][-100:]

        # If no existing tracks, just create new tracks for all detections
        if not self.tracks:
            for det in detections:
                self._initiate_track(det)
            return list(self.tracks.values())

        # If no detections, age all tracks and return
        if not detections:
            for track in self.tracks.values():
                track.age += 1
            # Remove old tracks
            self.tracks = {
                tid: t for tid, t in self.tracks.items()
                if t.age <= self.max_age
            }
            return list(self.tracks.values())

        # Build track and detection arrays
        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_bboxes = np.array([d.bbox for d in detections])

        # Compute IoU matrix
        iou_matrix = self.iou_batch(track_bboxes, det_bboxes)

        # Hungarian matching on IoU
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Process matches
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                det = detections[c]
                self._update_track(tid, det)
                matched_tracks.add(r)
                matched_dets.add(c)

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._initiate_track(det)

        # Remove old unmatched tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age
        }

        return list(self.tracks.values())

    def _initiate_track(self, detection) -> None:
        tid = self._next_track_id()
        track = Track(
            track_id=tid,
            bbox=detection.bbox,
            score=detection.score,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age=0,
            is_confirmed=True,
            feature=detection.feature,
            trajectory=[detection.center.tolist()]
        )
        self.tracks[tid] = track
        self.feature_cache[tid] = [detection.feature] if detection.feature is not None else []

    def _update_track(self, tid: int, detection) -> None:
        track = self.tracks[tid]
        track.bbox = detection.bbox
        track.score = detection.score
        track.class_id = detection.class_id
        track.is_confirmed = True
        track.age = 0
        track.trajectory.append(detection.center.tolist())
        if detection.feature is not None:
            track.feature = detection.feature