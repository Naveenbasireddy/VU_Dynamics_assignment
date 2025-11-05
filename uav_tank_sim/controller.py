
import math

class UAVController:
    def __init__(self, altitude=10.0, max_speed=0.12, hit_threshold=0.6):
        self.altitude = altitude
        self.max_speed = max_speed  # m per frame in our sim
        self.hit_threshold = hit_threshold

    def compute_velocity(self, drone_world_pos, target_world_pos):
        """
        drone_world_pos: np.array([x,y]) in meters
        target_world_pos: (x,y) in meters
        returns (vx, vy) per frame (meters)
        """
        if target_world_pos is None:
            return 0.0, 0.0
        dx = target_world_pos[0] - drone_world_pos[0]
        dy = target_world_pos[1] - drone_world_pos[1]
        dist = math.hypot(dx, dy) + 1e-6
        if dist < 1e-3:
            return 0.0, 0.0
        scale = min(self.max_speed, dist)
        vx = (dx / dist) * scale
        vy = (dy / dist) * scale
        return vx, vy
