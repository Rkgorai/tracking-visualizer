from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track
from trackers.traditional.kalman_filter import KalmanFilter


class OCSortTracker(BaseTracker):
    """OC-SORT: Observation-Centric SORT."""

    def __init__(
        self,
        det_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        **kwargs
    ):
        super().__init__(name="OC-SORT", **kwargs)
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.asso_func = asso_func

        self.observations: dict[int, list] = {}  # Track observations history
        self.lost_tracks: dict[int, Track] = {}

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        # Update age of existing tracks
        for track in self.tracks.values():
            track.age += 1
            if track.track_id not in self.observations:
                self.observations[track.track_id] = []
            self.observations[track.track_id].append(track.center.tolist())

        # Filter by threshold
        valid_dets = [d for d in detections if d.score >= self.det_thresh]

        # First: match existing tracks with detections
        matched, unmatched_tracks, unmatched_dets = self._match_tracks(valid_dets)

        # Update matched tracks
        for trk_idx, det_idx in matched:
            tid = list(self.tracks.keys())[trk_idx]
            det = valid_dets[det_idx]
            self._update_track(tid, det)

        # Second: match unmatched tracks with unmatched detections using lower threshold
        if unmatched_tracks and unmatched_dets:
            remaining_bboxes = np.array([self.tracks[tid].bbox for tid in unmatched_tracks])
            remaining_dets = [valid_dets[i] for i in unmatched_dets]
            det_bboxes = np.array([d.bbox for d in remaining_dets])

            iou_matrix = self.iou_batch(remaining_bboxes, det_bboxes)
            matched2, _, unmatched_dets2 = self._match(iou_matrix, 0.3)

            for trk_idx, det_idx in matched2:
                tid = list(unmatched_tracks)[trk_idx]
                det = remaining_dets[det_idx]
                self._update_track(tid, det)

            # Create new tracks for remaining unmatched detections
            for det_idx in unmatched_dets2:
                self._initiate_track(remaining_dets[det_idx])
        else:
            for det_idx in unmatched_dets:
                self._initiate_track(valid_dets[det_idx])

        # Move unmatched tracks to lost
        for trk_idx in unmatched_tracks:
            tid = list(self.tracks.keys())[trk_idx]
            self.lost_tracks[tid] = self.tracks.pop(tid)

        # Prune old lost tracks
        self.lost_tracks = {
            tid: t for tid, t in self.lost_tracks.items()
            if self.frame_count - t.age < self.max_age
        }

        # Remove old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age
        }

        # Return confirmed tracks
        return [t for t in self.tracks.values() if t.age >= self.min_hits]

    def _match_tracks(self, detections: list) -> tuple:
        if not self.tracks or not detections:
            unmatched_tracks = list(self.tracks.keys())
            unmatched_dets = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_dets

        track_bboxes = np.array([t.bbox for t in self.tracks.values()])
        det_bboxes = np.array([d.bbox for d in detections])

        iou_matrix = self.iou_batch(track_bboxes, det_bboxes)

        # Apply OCR (Observation-Centric Recovery)
        for track_id, obs in self.observations.items():
            if len(obs) < self.delta_t:
                continue
            # Direction-aware matching
            recent_obs = np.array(obs[-self.delta_t:])
            if len(recent_obs) >= 2:
                direction = recent_obs[-1] - recent_obs[0]
                for i, det in enumerate(detections):
                    det_center = det.center
                    if len(recent_obs) > 0:
                        pred_center = recent_obs[-1] + direction
                        dir_diff = det_center - pred_center
                        if np.linalg.norm(dir_diff) < 50:  # Direction threshold
                            iou_matrix[track_id, i] *= 1.2  # Boost IoU

        return self._match(iou_matrix, self.iou_threshold)

    def _match(self, iou_matrix: np.ndarray, threshold: float):
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(iou_matrix.shape[0]))
        unmatched_dets = set(range(iou_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= threshold:
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, list(unmatched_tracks), list(unmatched_dets)

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
        self.observations[tid] = [detection.center.tolist()]

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