import cv2
import numpy as np
from tracker import SimpleTracker
import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUT_DIR, "demo_output.mp4")

WIDTH, HEIGHT = 640, 360
FPS = 20
DURATION = 10
FRAME_COUNT = DURATION * FPS


def generate_positions(frame_idx):
    t = frame_idx / FPS
    if t < 2.5:
        ax = int(100 + 60 * t)
        ay = 120
        bx = int(500 - 60 * t)
        by = 120
    elif 2.5 <= t < 5.0:
        ax, ay = None, None
        bx, by = None, None
    else:
        t2 = t - 5.0
        ax = int(320 - 80 * t2)
        ay = 120
        bx = int(320 + 80 * t2)
        by = 120
    return (ax, ay), (bx, by)


def draw_object(frame, center, obj_id_hint=None):
    if center is None or center[0] is None or center[1] is None:
        return
    x, y = center
    w, h = 40, 25
    cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (200, 200, 200), -1)
    cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (60, 60, 60), 2)
    if obj_id_hint is not None:
        cv2.putText(frame, f"H{obj_id_hint}", (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    tracker = SimpleTracker()

    for i in range(FRAME_COUNT):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        posA, posB = generate_positions(i)

        draw_object(frame, posA, obj_id_hint="A")
        draw_object(frame, posB, obj_id_hint="B")

        detections = []
        if posA[0] is not None:
            x, y = posA
            detections.append(((x - 20, y - 12, 40, 25), None))
        if posB[0] is not None:
            x, y = posB
            detections.append(((x - 20, y - 12, 40, 25), None))

        tracked = tracker.update([d[0] for d in detections], frame_id=i)

        for tr in tracked:
            tid, bbox = tr['id'], tr['bbox']
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 120, 255), 2)
            cv2.putText(frame, f"ID:{tid}", (int(x), int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            pts = tracker.get_history(tid)
            for p in pts:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        cv2.putText(frame, f"Frame {i}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Demo video saved to {VIDEO_PATH}")
    print("Tracker log:")
    tracker.print_log()


if __name__ == "__main__":
    main()
