"""
main.py
UAV (drone) simulated demo for Assignment Q2:
- Drone at 10m altitude searches for a ground target (tank),
- When detected, drone approaches and registers a simulated 'hit' (proximity),
- Demonstrates temporary occlusion and re-identification logic.
Output: outputs/demo_q2.mp4 and console logs.

Run:
    python main.py
"""
import cv2
import numpy as np
import os
from detector import SimulatedDetector
from tracker import KalmanCentroidTracker
from controller import UAVController

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUT_DIR, "demo_q2.mp4")

WIDTH, HEIGHT = 800, 480
FPS = 20
DURATION = 12  # seconds
FRAME_COUNT = DURATION * FPS

def draw_scene(frame, drone_pos, target_pos, detected_bbox, track_id, status_text):
    # background ground
    cv2.rectangle(frame, (0, HEIGHT//2), (WIDTH, HEIGHT), (34,139,34), -1)
    # sky
    cv2.rectangle(frame, (0, 0), (WIDTH, HEIGHT//2), (135,206,235), -1)

    # draw target (tank) as a rectangle on ground (if visible)
    if target_pos is not None:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        w, h = 120, 50
        cv2.rectangle(frame, (tx-w//2, ty-h//2), (tx+w//2, ty+h//2), (50,50,50), -1)
        cv2.putText(frame, "Tank", (tx-30, ty-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # draw drone (top-down marker projected)
    if drone_pos is not None:
        dx, dy = int(drone_pos[0]), int(drone_pos[1])
        # drone represented by circle at its projected ground position
        cv2.circle(frame, (dx, dy), 12, (0,0,200), -1)
        cv2.putText(frame, "Drone (10m)", (dx-40, dy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # draw detection bbox if present
    if detected_bbox is not None:
        x,y,w,h = detected_bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,200,200), 2)
        cv2.putText(frame, f"Det (frame)", (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 2)

    # draw tracking ID if found
    if track_id is not None:
        cv2.putText(frame, f"Track ID: {track_id}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    # overlay status
    cv2.putText(frame, status_text, (10, HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)


def world_to_screen(x, y):
    # world coords in meters -> screen pixels for this simple sim
    # assume world x in [-6, 6] maps to width, y in [-4,4] to ground area
    sx = int(WIDTH/2 + x * (WIDTH/12))
    sy = int(HEIGHT*0.65 + y * (HEIGHT/8))
    return sx, sy

def generate_target_position(t):
    # target moves slowly in an oval path; returns world coords (x,y)
    x = 3.0 * np.cos(0.35 * t)
    y = 1.8 * np.sin(0.25 * t)
    return x, y

def main():
    # initialize components
    detector = SimulatedDetector()
    tracker = KalmanCentroidTracker(max_age=40)
    controller = UAVController(altitude=10.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    # drone initial projected ground position (start over center)
    drone_world = np.array([0.0, -1.5])  # x,y in meters
    drone_speed = 0.15  # m per frame (scaled)

    hit_logged = False

    for i in range(FRAME_COUNT):
        t = i / FPS
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # target motion, and simulate occlusion: hide for 3 seconds every 9 seconds
        target_visible = (int(t) % 9) < 6
        target_world = generate_target_position(t) if target_visible else None

        # convert to screen coords
        target_screen = world_to_screen(*target_world) if target_world is not None else None
        drone_screen = world_to_screen(*drone_world)

        # Detector: returns bbox in screen coords if visible
        det = detector.detect(target_screen, visible=target_visible)

        # Tracker: update with detection (if any) and get track id + smoothed position
        tracked = tracker.update([det]) if det is not None else tracker.update([])
        track_id = None
        tracked_pos = None
        if tracked:
            # tracker returns list of tracks with centroids in screen coords
            tr = tracked[0]
            track_id = tr['id']
            tracked_pos = tr['centroid']

        # Controller: if detection exists or we have a recent track, compute velocity toward target
        status = "Searching..."
        if det is not None or tracked_pos is not None:
            # choose target position: if detection available use its centroid; else use tracker prediction
            if det is not None:
                tx = det[0] + det[2]/2.0
                ty = det[1] + det[3]/2.0
            else:
                tx, ty = tracked_pos
            # convert screen -> world coordinates approximate inverse
            # invert world_to_screen roughly:
            wx = (tx - WIDTH/2) / (WIDTH/12)
            wy = (ty - int(HEIGHT*0.65)) / (HEIGHT/8)
            # controller computes a movement vector (ground plane)
            vx, vy = controller.compute_velocity(drone_world, (wx, wy))
            # apply movement
            drone_world += np.array([vx, vy])
            status = "Approaching target"
            # check hit: distance in world meters
            if target_world is not None:
                dist = np.hypot(drone_world[0]-target_world[0], drone_world[1]-target_world[1])
                if dist < controller.hit_threshold and not hit_logged:
                    print(f"[{i}] HIT EVENT: drone reached target proximity (dist={dist:.2f} m)")
                    hit_logged = True
        else:
            # search motion: simple slow circular drift
            drone_world[0] += 0.02 * np.cos(0.5 * t)
            drone_world[1] += 0.02 * np.sin(0.5 * t)
            status = "Searching (no detection)"

        # draw everything
        draw_scene(frame, drone_screen, target_screen, det, track_id, status)
        writer.write(frame)

    writer.release()
    print("Demo saved to", VIDEO_PATH)
    print("Tracker log (recent):")
    tracker.print_log()

if __name__ == "__main__":
    main()
