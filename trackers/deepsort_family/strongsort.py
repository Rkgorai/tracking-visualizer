import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.base import BaseTracker, Track
from trackers.deepsort_family.deepsort import DeepSORT


class StrongSORT(DeepSORT):
    """
    StrongSORT: StrongSORT with Global Search and Identification (GSI),
    Appearance Feature Integration (AFI), and Motion Consistency (MC).
    """

    def __init__(
        self,
        ema_alpha: float = 0.9,
        gsi_threshold: float = 0.5,
        mc_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = "StrongSORT"
        self.ema_alpha = ema_alpha  # EMA for feature smoothing
        self.gsi_threshold = gsi_threshold
        self.mc_threshold = mc_threshold

    def update(self, detections: list) -> list[Track]:
        self.frame_count += 1

        # Update age and EMA features
        for track in self.tracks.values():
            track.age += 1
            if track.feature is not None:
                if not hasattr(track, 'ema_feature') or track.ema_feature is None:
                    track.ema_feature = track.feature.copy()
                else:
                    # EMA update
                    track.ema_feature = (
                        self.ema_alpha * track.feature +
                        (1 - self.ema_alpha) * track.ema_feature
                    )

        # GSI - Global Search and Identification
        matched, unmatched_tracks, unmatched_dets = self._gsi_matching(detections)

        # Update matched tracks
        for trk_idx, det_idx in matched:
            tid = list(self.tracks.keys())[trk_idx]
            det = detections[det_idx]
            self._update_track(tid, det)

        # MC - Motion Consistency matching for unmatched
        if unmatched_tracks and unmatched_dets:
            mc_matched = self._mc_matching(unmatched_tracks, unmatched_dets, detections)

            for trk_idx, det_idx in mc_matched:
                tid = unmatched_tracks[trk_idx]
                det = detections[unmatched_dets[det_idx]]
                self._update_track(tid, det)

                unmatched_tracks.remove(tid)
                unmatched_dets.remove(det_idx)

        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])

        # Mark unmatched tracks as lost
        for trk_idx in unmatched_tracks:
            tid = list(self.tracks.keys())[trk_idx]
            if tid in self.tracks:
                self.tracks[tid].is_confirmed = False

        # Prune old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age <= self.max_age and (t.is_confirmed or t.age < self.n_init)
        }

        return [t for t in self.tracks.values()]

    def _gsi_matching(self, detections: list) -> tuple:
        """Global Search and Identification matching."""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))) if self.tracks else [], list(range(len(detections))) if detections else []

        # Build cost matrix with appearance and motion
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks.values()):
            # Motion cost (IoU)
            track_bbox = track.bbox
            for j, det in enumerate(detections):
                iou = self.iou(track_bbox, det.bbox)
                motion_cost = 1 - iou

                # Appearance cost (cosine distance)
                app_cost = 0.0
                ema_feature = getattr(track, 'ema_feature', None)
                if ema_feature is not None and det.feature is not None:
                    app_cost = 1 - np.dot(ema_feature, det.feature) / (
                        np.linalg.norm(ema_feature) * np.linalg.norm(det.feature) + 1e-7
                    )

                # Combined cost
                cost_matrix[i, j] = 0.5 * motion_cost + 0.5 * app_cost

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.6:  # Threshold
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _mc_matching(self, track_indices: list, det_indices: list, detections: list) -> list:
        """Motion Consistency matching."""
        matched = []

        track_list = [list(self.tracks.values())[i] for i in track_indices]

        for i, track in enumerate(track_list):
            if len(track.trajectory) < 2:
                continue

            # Predict next position from trajectory
            recent = track.trajectory[-3:]
            if len(recent) >= 2:
                velocity = np.array(recent[-1]) - np.array(recent[0])
                predicted = np.array(recent[-1]) + velocity

                # Find detection closest to prediction
                best_idx = None
                best_dist = float('inf')

                for j, det_idx in enumerate(det_indices):
                    det = detections[det_idx]
                    dist = np.linalg.norm(det.center - predicted)
                    if dist < best_dist and dist < 50:
                        best_dist = dist
                        best_idx = j

                if best_idx is not None:
                    matched.append((i, best_idx))

        return matched