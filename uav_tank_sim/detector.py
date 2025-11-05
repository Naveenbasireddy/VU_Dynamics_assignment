
import random
import math

class SimulatedDetector:
    def __init__(self):
        pass

    def detect(self, target_screen, visible=True):
   
        if not visible or target_screen is None:
            return None
        sx, sy = target_screen
        w, h = 120, 50
        jitter = (random.uniform(-4,4), random.uniform(-2,2))
        x = sx - w//2 + jitter[0]
        y = sy - h//2 + jitter[1]
        return (int(x), int(y), int(w), int(h))
