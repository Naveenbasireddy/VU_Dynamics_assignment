# detector.py
# Simulated detector that returns a bounding box (x,y,w,h) in screen pixels when the target is visible.
import random
import math

class SimulatedDetector:
    def __init__(self):
        pass

    def detect(self, target_screen, visible=True):
        """
        target_screen: (sx,sy) center of target in screen coords or None
        visible: bool whether target is in view
        returns bbox (x,y,w,h) or None
        """
        if not visible or target_screen is None:
            return None
        sx, sy = target_screen
        # bbox size depends on distance (we keep fixed for simplicity)
        w, h = 120, 50
        jitter = (random.uniform(-4,4), random.uniform(-2,2))
        x = sx - w//2 + jitter[0]
        y = sy - h//2 + jitter[1]
        return (int(x), int(y), int(w), int(h))
