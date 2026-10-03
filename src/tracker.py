"""Lightweight IoU tracker with constant-velocity prediction.
predict() moves boxes on skipped frames; update() associates detections (Hungarian)."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list
    conf: float
    hits: int = 1
    missed: int = 0                  # consecutive detection cycles without a match
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    face_id: int = None
    entered: bool = False
    id_attempts: int = 0
    updated: bool = False            # matched in the current detection cycle
    last_crop: np.ndarray = None
    best_area: float = 0.0

    @property
    def size(self):
        return min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


class IoUTracker:
    def __init__(self, iou_thr=0.25, max_missed=6):
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks, self._next = [], 1

    def predict(self):
        for t in self.tracks:
            dx, dy = t.vel
            t.bbox = [t.bbox[0] + dx, t.bbox[1] + dy, t.bbox[2] + dx, t.bbox[3] + dy]
            t.vel = t.vel * 0.9

    def update(self, dets, frames_since_last=1):
        """dets: [(x1,y1,x2,y2,conf)]. Returns tracks that expired this cycle."""
        for t in self.tracks:
            t.updated = False
        matched_d = set()
        if self.tracks and dets:
            cost = np.array([[1 - iou(t.bbox, d[:4]) for d in dets] for t in self.tracks])
            for r, c in zip(*linear_sum_assignment(cost)):
                if 1 - cost[r, c] < self.iou_thr:
                    continue
                t, d = self.tracks[r], dets[c]
                old_c = np.array([(t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2])
                new_c = np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])
                t.vel = 0.5 * (new_c - old_c) / max(1, frames_since_last) + 0.5 * t.vel
                t.bbox, t.conf = list(d[:4]), d[4]
                t.hits += 1
                t.missed, t.updated = 0, True
                matched_d.add(c)
        for i, d in enumerate(dets):
            if i not in matched_d:
                self.tracks.append(Track(self._next, list(d[:4]), d[4], updated=True))
                self._next += 1
        lost = []
        for t in list(self.tracks):
            if not t.updated:
                t.missed += 1
                if t.missed > self.max_missed:
                    lost.append(t)
                    self.tracks.remove(t)
        return lost
