from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track
from trackers.traditional.kalman_filter import KalmanFilter


class BoTSORTracker(BaseTracker):
    """BoT-SORT: Robust Associations by Re-Identification in Multi-Object Tracking."""

    def __init__(
        self,
        reid_model=None,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_reid: bool = True,
        lambda_: float = 0.98,
        **kwargs
    ):
        super().__init__(name="BoT-SORT", **kwargs)
        self.reid_model = reid_model
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_reid = use_reid
        self.lambda_ = lambda_

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        for track in self.tracks.values():
            track.age += 1

        # First association: IoU matching
        if self.tracks and detections:
            track_bboxes = np.array([t.bbox for t in self.tracks.values()])
            det_bboxes = np.array([d.bbox for d in detections])
            iou_matrix = self.iou_batch(track_bboxes, det_bboxes)

            # Apply camera motion compensation if needed
            cost_matrix = 1 - iou_matrix

            # First matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched = []
            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_dets = set(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

            # Update matched
            for trk_idx, det_idx in matched:
                tid = list(self.tracks.keys())[trk_idx]
                det = detections[det_idx]
                self._update_track(tid, det)

            # Second matching with ReID (if available)
            if self.use_reid and unmatched_tracks and unmatched_dets:
                unmatched_track_ids = [list(self.tracks.keys())[i] for i in unmatched_tracks]
                unmatched_dets_list = [detections[i] for i in unmatched_dets]

                reid_matches = self._reid_matching(unmatched_track_ids, unmatched_dets_list)

                for tid, det in reid_matches:
                    self._update_track(tid, det)
                    unmatched_tracks.discard(list(self.tracks.keys()).index(tid))
                    unmatched_dets.discard(detections.index(det))

            # Mark unmatched as unconfirmed
            for trk_idx in unmatched_tracks:
                tid = list(self.tracks.keys())[trk_idx]
                self.tracks[tid].is_confirmed = False

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._initiate_track(detections[det_idx])
        else:
            for det in detections:
                self._initiate_track(det)

        # Remove old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age and (t.is_confirmed or t.age < self.min_hits)
        }

        return list(self.tracks.values())

    def _reid_matching(self, track_ids: list, detections: list) -> list:
        """ReID appearance matching."""
        if not self.reid_model or not detections:
            return []

        # Extract features for detections
        det_features = []
        for det in detections:
            if det.feature is not None:
                det_features.append(det.feature)
            else:
                det_features.append(np.zeros(128))

        matched = []
        for tid in track_ids:
            track = self.tracks.get(tid)
            if track and track.feature is not None:
                # Find best matching detection
                best_idx = None
                best_score = -1

                for idx, feat in enumerate(det_features):
                    if feat is not None:
                        # Cosine similarity
                        score = np.dot(track.feature, feat) / (
                            np.linalg.norm(track.feature) * np.linalg.norm(feat) + 1e-7
                        )
                        if score > best_score and score > 0.3:
                            best_score = score
                            best_idx = idx

                if best_idx is not None:
                    matched.append((tid, detections[best_idx]))

        return matched

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