from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track
from trackers.traditional.kalman_filter import KalmanFilter


class ByteTrack(BaseTracker):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        lost_track_buffer: int = 30,
        min_box_area: float = 10,
        track_iou_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(name="ByteTrack", **kwargs)
        self.track_thresh = track_thresh
        self.lost_track_buffer = lost_track_buffer
        self.min_box_area = min_box_area
        self.track_iou_threshold = track_iou_threshold

        self.lost_tracks: dict[int, Track] = {}

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        # Split detections by confidence
        high_dets = [d for d in detections if d.score >= self.track_thresh]
        low_dets = [d for d in detections if 0.1 <= d.score < self.track_thresh]

        # Update existing tracks with high confidence detections
        remaining_tracks = self._update_active_tracks(high_dets)

        # Try to recover lost tracks with high confidence detections
        self._recover_lost_tracks(remaining_tracks, high_dets)

        # Match remaining tracks with low confidence detections
        self._match_remaining_with_low(remaining_tracks, low_dets)

        # Mark unmatched tracks as lost
        for tid in remaining_tracks:
            if tid in self.tracks:
                self.lost_tracks[tid] = self.tracks.pop(tid)

        # Buffer lost tracks
        self.lost_tracks = {
            tid: t for tid, t in self.lost_tracks.items()
            if self.frame_count - t.age < self.lost_track_buffer
        }

        # Remove very old lost tracks
        all_tracks = {**self.tracks, **self.lost_tracks}
        all_tracks = {
            tid: t for tid, t in all_tracks.items()
            if t.age < self.lost_track_buffer
        }

        return list(self.tracks.values())

    def _update_active_tracks(self, detections: list) -> set[int]:
        """Update active tracks with high confidence detections."""
        remaining_tracks = set(self.tracks.keys())

        if not detections or not self.tracks:
            return remaining_tracks

        det_bboxes = np.array([d.bbox for d in detections])

        # First match: high IoU
        for track in self.tracks.values():
            track.age += 1

        track_bboxes = np.array([t.bbox for t in self.tracks.values()])
        iou_matrix = self.iou_batch(track_bboxes, det_bboxes)

        matched, unmatched_dets, unmatched_tracks = self._match(
            iou_matrix, self.track_iou_threshold
        )

        for trk_idx, det_idx in matched:
            tid = list(self.tracks.keys())[trk_idx]
            det = detections[det_idx]
            self._update_track(tid, det)
            remaining_tracks.discard(tid)

        # Second match: low IoU for unmatched tracks
        if unmatched_tracks and unmatched_dets:
            unmatched_track_bboxes = np.array([self.tracks[tid].bbox for tid in remaining_tracks])
            unmatched_det_bboxes = np.array([detections[i].bbox for i in unmatched_dets])

            if len(unmatched_track_bboxes) > 0 and len(unmatched_det_bboxes) > 0:
                iou_matrix2 = self.iou_batch(unmatched_track_bboxes, unmatched_det_bboxes)
                matched2, _, _ = self._match(iou_matrix2, 0.3)

                for trk_idx, det_idx2 in matched2:
                    tid = list(remaining_tracks)[trk_idx]
                    det = detections[unmatched_dets[det_idx2]]
                    self._update_track(tid, det)
                    remaining_tracks.discard(tid)

        # Initiate new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det.area >= self.min_box_area:
                self._initiate_track(det)

        return remaining_tracks

    def _recover_lost_tracks(self, remaining_tracks: set[int], detections: list) -> None:
        """Try to recover lost tracks with high confidence detections."""
        if not self.lost_tracks:
            return

        lost_bboxes = np.array([t.bbox for t in self.lost_tracks.values()])
        det_bboxes = np.array([d.bbox for d in detections])

        if len(lost_bboxes) > 0 and len(det_bboxes) > 0:
            iou_matrix = self.iou_batch(lost_bboxes, det_bboxes)
            matched, _, _ = self._match(iou_matrix, 0.4)

            for lost_idx, det_idx in matched:
                lost_id = list(self.lost_tracks.keys())[lost_idx]
                det = detections[det_idx]
                self._initiate_track(det, track_id=lost_id)
                del self.lost_tracks[lost_id]

    def _match_remaining_with_low(self, remaining_tracks: set[int], low_dets: list) -> None:
        """Match remaining tracks with low confidence detections."""
        if not remaining_tracks or not low_dets:
            return

        track_bboxes = np.array([self.tracks[tid].bbox for tid in remaining_tracks])
        det_bboxes = np.array([d.bbox for d in low_dets])

        iou_matrix = self.iou_batch(track_bboxes, det_bboxes)
        matched, _, _ = self._match(iou_matrix, 0.5)

        for trk_idx, det_idx in matched:
            tid = list(remaining_tracks)[trk_idx]
            det = low_dets[det_idx]
            self._update_track(tid, det)
            remaining_tracks.discard(tid)

    def _match(self, iou_matrix: np.ndarray, threshold: float):
        """Match using Hungarian algorithm."""
        if iou_matrix.size == 0:
            return [], [], list(range(iou_matrix.shape[0]))

        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(iou_matrix.shape[1]))
        unmatched_tracks = set(range(iou_matrix.shape[0]))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= threshold:
                matched.append((r, c))
                unmatched_dets.discard(c)
                unmatched_tracks.discard(r)

        return matched, list(unmatched_dets), list(unmatched_tracks)

    def _initiate_track(self, detection, track_id: Optional[int] = None) -> None:
        tid = track_id if track_id is not None else self._next_track_id()
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