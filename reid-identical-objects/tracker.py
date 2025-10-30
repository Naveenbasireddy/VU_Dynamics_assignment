

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import math

TRACK_MAX_AGE = 30  
IOU_THRESHOLD = 1e9  

def centroid_from_bbox(bbox):
    x,y,w,h = bbox
    return (x + w/2.0, y + h/2.0)

class Track:
    def __init__(self, tid, bbox, frame_id):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame_id
        self.age = 0
        self.missing = 0
        self.history = deque(maxlen=200)
        c = centroid_from_bbox(bbox)
        self.history.append(c)
        self.vx, self.vy = 0.0, 0.0

    def predict(self, dt=1.0):
        cx, cy = self.history[-1]
        return (cx + self.vx * dt, cy + self.vy * dt)

    def update(self, bbox, frame_id):
        new_c = centroid_from_bbox(bbox)
        if len(self.history) >= 1:
            old_c = self.history[-1]
            self.vx = new_c[0] - old_c[0]
            self.vy = new_c[1] - old_c[1]
        self.history.append(new_c)
        self.bbox = bbox
        self.last_frame = frame_id
        self.missing = 0

    def mark_missed(self):
        self.missing += 1

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = []
        self.log = []

    def update(self, detections, frame_id):
        det_centroids = [centroid_from_bbox(b) for b in detections]

        preds = [t.predict() for t in self.tracks]

        if len(self.tracks) == 0:
            for d_bbox in detections:
                t = Track(self.next_id, d_bbox, frame_id)
                self.tracks.append(t)
                self.log.append((frame_id, "init", t.id))
                self.next_id += 1
            return [{"id": t.id, "bbox": t.bbox} for t in self.tracks]

        if len(preds) > 0 and len(det_centroids) > 0:
            cost = np.zeros((len(preds), len(det_centroids)), dtype=float)
            for i,p in enumerate(preds):
                for j,d in enumerate(det_centroids):
                    cost[i,j] = math.hypot(p[0]-d[0], p[1]-d[1])
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        assigned_tracks = set()
        assigned_dets = set()
        for r,c in zip(row_ind, col_ind):
            # Assign detection c to track r
            self.tracks[r].update(detections[c], frame_id)
            assigned_tracks.add(r)
            assigned_dets.add(c)
            self.log.append((frame_id, "match", self.tracks[r].id, detections[c]))

        for i,t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.mark_missed()
                self.log.append((frame_id, "missed", t.id, t.missing))
        for j,d_bbox in enumerate(detections):
            if j not in assigned_dets:
                newt = Track(self.next_id, d_bbox, frame_id)
                self.tracks.append(newt)
                self.log.append((frame_id, "spawn", newt.id))
                self.next_id += 1

        alive = []
        for t in self.tracks:
            if t.missing <= TRACK_MAX_AGE:
                alive.append(t)
            else:
                self.log.append((frame_id, "delete", t.id))
        self.tracks = alive

        return [{"id": t.id, "bbox": t.bbox} for t in self.tracks]

    def get_history(self, tid):
        for t in self.tracks:
            if t.id == tid:
                return list(t.history)
        return []

    def print_log(self):
        for e in self.log:
            print(e)
