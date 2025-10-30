# tracker.py
# Simple centroid tracker with a Kalman-like smoothing for centroid positions.
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from collections import deque

class Track:
    def __init__(self, tid, centroid):
        self.id = tid
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        # state: x, y, vx, vy
        self.kf.F = np.array([[1,0,dt,0],
                              [0,1,0,dt],
                              [0,0,1,0],
                              [0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0]])
        self.kf.R *= 10.0
        self.kf.P *= 100.0
        self.kf.Q *= 0.01
        self.kf.x = np.array([centroid[0], centroid[1], 0., 0.])
        self.age = 0
        self.missing = 0
        self.history = deque(maxlen=200)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, centroid):
        z = np.array([centroid[0], centroid[1]])
        self.kf.update(z)
        self.age += 1
        self.missing = 0
        self.history.append(self.kf.x[:2])

    def get_state(self):
        return self.kf.x[:2]

class KalmanCentroidTracker:
    def __init__(self, max_age=30, dist_thresh=80):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.log = []

    def update(self, detections):
        """
        detections: list of bbox tuples (x,y,w,h) OR [].
        returns list of track dicts: [{'id': id, 'centroid': (x,y)}]
        """
        centroids = []
        for d in detections:
            if d is None:
                continue
            x,y,w,h = d
            centroids.append((x + w/2.0, y + h/2.0))

        # Predict existing
        preds = []
        for tr in self.tracks:
            pred = tr.predict()
            preds.append(pred)

        assigned = {}
        if len(preds) == 0 and len(centroids) > 0:
            # create tracks for all detections
            for c in centroids:
                tr = Track(self.next_id, c)
                self.tracks.append(tr)
                self.log.append(("init", tr.id))
                self.next_id += 1
        elif len(preds) > 0 and len(centroids) > 0:
            # cost matrix
            cost = np.zeros((len(preds), len(centroids)))
            for i,p in enumerate(preds):
                for j,c in enumerate(centroids):
                    cost[i,j] = distance.euclidean(p, c)
            # greedy assignment
            for _ in range(min(cost.shape)):
                i,j = np.unravel_index(cost.argmin(), cost.shape)
                if cost[i,j] > self.dist_thresh:
                    break
                # assign
                self.tracks[i].update(centroids[j])
                assigned[i] = j
                cost[i,:] = 1e6
                cost[:,j] = 1e6

            # unmatched detections -> spawn new tracks
            matched_dets = set(assigned.values())
            for idx, c in enumerate(centroids):
                if idx not in matched_dets:
                    tr = Track(self.next_id, c)
                    self.tracks.append(tr)
                    self.log.append(("spawn", tr.id))
                    self.next_id += 1

            # unmatched tracks -> mark missing
            for i,tr in enumerate(self.tracks):
                if i not in assigned:
                    tr.missing += 1
        else:
            # no detections -> mark all missing
            for tr in self.tracks:
                tr.missing += 1

        # remove dead tracks
        alive = []
        for tr in self.tracks:
            if tr.missing <= self.max_age:
                alive.append(tr)
            else:
                self.log.append(("del", tr.id))
        self.tracks = alive

        # prepare return
        out = []
        for tr in self.tracks:
            pos = tr.get_state()
            out.append({'id': tr.id, 'centroid': (float(pos[0]), float(pos[1]))})
        return out

    def print_log(self):
        for e in self.log[-100:]:
            print(e)
