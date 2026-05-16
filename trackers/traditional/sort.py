from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track
from trackers.traditional.kalman_filter import KalmanFilter


class SortTracker(BaseTracker):
    """Simple Online and Realtime Tracking with Kalman Filter and Hungarian algorithm."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3, **kwargs):
        super().__init__(name="SORT", **kwargs)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        # Get predicted tracks
        for track in self.tracks.values():
            track.age += 1

        # Associate detections with existing tracks
        if self.tracks:
            track_bboxes = np.array([t.bbox for t in self.tracks.values()])
            det_bboxes = np.array([d.bbox for d in detections]) if detections else np.zeros((0, 4))

            if detections:
                iou_matrix = self.iou_batch(track_bboxes, det_bboxes)
                matched, unmatched_dets, unmatched_tracks = self._match(
                    iou_matrix, self.iou_threshold
                )

                for trk_idx, det_idx in matched:
                    tid = list(self.tracks.keys())[trk_idx]
                    det = detections[det_idx]
                    self._update_track(tid, det)

                for det_idx in unmatched_dets:
                    self._initiate_track(detections[det_idx])

                for trk_idx in unmatched_tracks:
                    tid = list(self.tracks.keys())[trk_idx]
                    self.tracks[tid].is_confirmed = False
            else:
                unmatched_tracks = list(range(len(self.tracks)))
                for trk_idx in unmatched_tracks:
                    tid = list(self.tracks.keys())[trk_idx]
                    self.tracks[tid].is_confirmed = False
        else:
            for det in detections:
                self._initiate_track(det)

        # Remove old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age and (t.age >= self.min_hits or t.is_confirmed)
        }

        return [t for t in self.tracks.values() if t.age >= self.min_hits]

    def _match(self, iou_matrix: np.ndarray, threshold: float):
        """Match using Hungarian algorithm on IoU matrix."""
        if iou_matrix.size == 0:
            return [], [], list(range(iou_matrix.shape[0]))

        iou_matrix = 1 - iou_matrix  # Convert to cost matrix
        row_ind, col_ind = linear_sum_assignment(iou_matrix)

        matched = []
        unmatched_dets = set(range(iou_matrix.shape[1]))
        unmatched_tracks = set(range(iou_matrix.shape[0]))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] <= 1 - threshold:
                matched.append((r, c))
                unmatched_dets.discard(c)
                unmatched_tracks.discard(r)

        return matched, list(unmatched_dets), list(unmatched_tracks)

    def _initiate_track(self, detection) -> None:
        tid = self._next_track_id()
        kf = KalmanFilter()
        kf.initiate(detection.bbox)
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
        track.kalman_filter = kf
        self.tracks[tid] = track

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