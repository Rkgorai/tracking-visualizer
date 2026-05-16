import numpy as np
from typing import Optional

from trackers.base import BaseTracker, Track


class MOTRTracker(BaseTracker):
    """
    MOTR: End-to-End Multi-Object Tracking with Transformer.

    This is a simplified implementation. The full MOTR requires
    a pretrained transformer model for detections and queries.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        query_embed_dim: int = 256,
        num_queries: int = 300,
        **kwargs
    ):
        super().__init__(name="MOTR", **kwargs)
        self.max_age = max_age
        self.min_hits = min_hits
        self.query_embed_dim = query_embed_dim
        self.num_queries = num_queries

        self.track_queries = {}  # Active track queries
        self.query_counter = 0

    def update(self, detections: list) -> list[Track]:
        """
        Update with MOTR-style processing.

        In the full implementation, this would:
        1. Encode frame features
        2. Process track queries through transformer decoder
        3. Predict bounding boxes for each query
        4. Associate with detections via bipartite matching
        """
        self.frame_count += 1

        # For simplified version, treat as detection-based tracker
        # with enhanced query management

        if not self.tracks:
            # First frame: create queries for all detections
            for det in detections:
                self._create_track_query(det)
        else:
            # Association via IoU + feature matching
            matched, unmatched_tracks, unmatched_dets = self._query_matching(detections)

            # Update matched tracks
            for trk_idx, det_idx in matched:
                tid = list(self.tracks.keys())[trk_idx]
                det = detections[det_idx]
                self._update_track_query(tid, det)

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track_query(detections[det_idx])

            # Update unmatched tracks
            for trk_idx in unmatched_tracks:
                tid = list(self.tracks.keys())[trk_idx]
                if tid in self.tracks:
                    self.tracks[tid].age += 1

        # Prune old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age and (t.is_confirmed or t.age < self.min_hits)
        }

        return [t for t in self.tracks.values() if t.is_confirmed or t.age >= self.min_hits]

    def _query_matching(self, detections: list) -> tuple:
        """Match track queries with detections."""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))) if self.tracks else [], list(range(len(detections))) if detections else []

        track_bboxes = np.array([t.bbox for t in self.tracks.values()])
        det_bboxes = np.array([d.bbox for d in detections])

        iou_matrix = self.iou_batch(track_bboxes, det_bboxes)

        # Add feature-based matching if available
        if any(t.feature is not None for t in self.tracks.values()) and \
           any(d.feature is not None for d in detections):
            for i, track in enumerate(self.tracks.values()):
                for j, det in enumerate(detections):
                    if track.feature is not None and det.feature is not None:
                        cosine_sim = np.dot(track.feature, det.feature) / (
                            np.linalg.norm(track.feature) * np.linalg.norm(det.feature) + 1e-7
                        )
                        iou_matrix[i, j] = 0.5 * iou_matrix[i, j] + 0.5 * (cosine_sim + 1) / 2

        # Simple greedy matching
        matched = []
        used_dets = set()

        for i in range(len(track_bboxes)):
            best_j = -1
            best_iou = 0.3
            for j in range(len(det_bboxes)):
                if j not in used_dets and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j

            if best_j >= 0:
                matched.append((i, best_j))
                used_dets.add(best_j)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in [m[0] for m in matched]]
        unmatched_dets = [j for j in range(len(detections)) if j not in [m[1] for m in matched]]

        return matched, unmatched_tracks, unmatched_dets

    def _create_track_query(self, detection) -> None:
        """Create new track query for a detection."""
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
        self.track_queries[tid] = {
            "query_features": detection.feature if detection.feature is not None else np.zeros(self.query_embed_dim),
            "num_updates": 0
        }

    def _update_track_query(self, tid: int, detection) -> None:
        """Update track query with new detection."""
        track = self.tracks[tid]
        track.bbox = detection.bbox
        track.score = detection.score
        track.class_id = detection.class_id
        track.is_confirmed = True
        track.age = 0
        track.trajectory.append(detection.center.tolist())

        if detection.feature is not None:
            track.feature = detection.feature

        # Update query state
        if tid in self.track_queries:
            q = self.track_queries[tid]
            # Exponential moving average of query features
            alpha = 0.9
            if detection.feature is not None:
                q["query_features"] = alpha * detection.feature + (1 - alpha) * q["query_features"]
            q["num_updates"] += 1