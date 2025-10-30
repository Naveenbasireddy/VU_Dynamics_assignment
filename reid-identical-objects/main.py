"""
main.py
Simulate two visually identical objects that exit frame, get swapped off-camera, then reappear.
Run a simple tracker (centroid + Kalman prediction) and show where identity switches happen.
Produces outputs/demo_output.mp4 and prints tracking events to console.

Usage:
    python main.py
"""

import cv2
import numpy as np
from tracker import SimpleTracker
import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUT_DIR, "demo_output.mp4")

WIDTH, HEIGHT = 640, 360
FPS = 20
DURATION = 10  # seconds
FRAME_COUNT = DURATION * FPS

def generate_positions(frame_idx):
    """Generate positions for two identical objects. They move, leave frame and get swapped off-camera."""
    t = frame_idx / FPS
    # object A: moves right, leaves at t=2.5s, re-enters at t=5.0s from swapped position
    # object B: moves left, leaves at t=2.5s, re-enters at t=5.0s from swapped position
    if t < 2.5:
        ax = int(100 + 60 * t)         # moves right
        ay = 120
        bx = int(500 - 60 * t)         # moves left
        by = 120
    elif 2.5 <= t < 5.0:
        # both off-frame: we simulate swap while off-screen by preparing re-entry positions
        ax, ay = None, None
        bx, by = None, None
    else:
        # reappear after swap: A appears where B would have reappeared and vice versa
        t2 = t - 5.0
        ax = int(320 - 80 * t2)  # reappear from right-to-left
        ay = 120
        bx = int(320 + 80 * t2)  # reappear from left-to-right
        by = 120
    return (ax, ay), (bx, by)

def draw_object(frame, center, obj_id_hint=None):
    if center is None or center[0] is None or center[1] is None:
        return  # skip drawing if object not visible
    x, y = center
    w, h = 40, 25
    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (200,200,200), -1)
    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (60,60,60), 2)
    if obj_id_hint is not None:
        cv2.putText(frame, f"H{obj_id_hint}", (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)

def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    tracker = SimpleTracker()

    for i in range(FRAME_COUNT):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        posA, posB = generate_positions(i)
        # draw hint labels for humans (these are not used by tracker)
        draw_object(frame, posA, obj_id_hint="A")
        draw_object(frame, posB, obj_id_hint="B")

        # Build detection list (only include if on-frame)
        detections = []
        if posA[0] is not None:
            x, y = posA
            detections.append(((x-20, y-12, 40, 25), None))  # bbox + None for feature
        if posB[0] is not None:
            x, y = posB
            detections.append(((x-20, y-12, 40, 25), None))

        # Tracker expects list of bboxes (x,y,w,h)
        tracked = tracker.update([d[0] for d in detections], frame_id=i)

        # Visualize tracked boxes and IDs
        for tr in tracked:
            tid, bbox = tr['id'], tr['bbox']
            x,y,w,h = bbox
            cv2.rectangle(frame, (int(x),int(y)), (int(x+w), int(y+h)), (0,120,255), 2)
            cv2.putText(frame, f"ID:{tid}", (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,120,255), 2)
            # draw trail
            pts = tracker.get_history(tid)
            for p in pts:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0,0,255), -1)

        # overlay info
        cv2.putText(frame, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        writer.write(frame)

    writer.release()
    print(f"Demo video saved to {VIDEO_PATH}")
    print("Tracker log:")
    tracker.print_log()

if __name__ == "__main__":
    main()
