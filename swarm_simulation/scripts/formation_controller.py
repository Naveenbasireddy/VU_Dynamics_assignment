#!/usr/bin/env python3
import rospy, math
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

class FormationController:
    def __init__(self):
        rospy.init_node('formation_controller')
        self.drone_id = int(rospy.get_param('~drone_id', 0))
        self.goal = None
        self.pose = {'x':0,'y':0,'z':rospy.get_param('/swarm_simulation/takeoff_alt',10.0)}
        self.pub_cmd = rospy.Publisher(f'/drone{self.drone_id}/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber(f'/drone{self.drone_id}/goal', PoseStamped, self.goal_cb)
        # For simplicity, read other goals (repulsion) via topics too (not robust but fine for demo)
        self.other_goals = {}
        for i in range(rospy.get_param('/swarm_simulation/num_drones',4)):
            rospy.Subscriber(f'/drone{i}/goal', PoseStamped, self.make_other_cb(i))
        rospy.Timer(rospy.Duration(0.1), self.control)
        rospy.loginfo("Controller %d started", self.drone_id)

    def make_other_cb(self, i):
        def cb(msg):
            self.other_goals[i] = msg.pose.position
        return cb

    def goal_cb(self, msg):
        self.goal = msg.pose.position

    def control(self, event):
        if self.goal is None:
            return
        # simple current pos = goal of last published for demo (we don't simulate physics); compute vector
        # For visual demo, we just publish moves toward goal so gazebo model can be controlled by plugin/bridge
        dx = self.goal.x - self.pose['x']
        dy = self.goal.y - self.pose['y']
        dz = self.goal.z - self.pose['z']
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
        max_speed = rospy.get_param('/swarm_simulation/max_speed', 1.5)
        vx = (dx/dist) * min(max_speed, dist)
        vy = (dy/dist) * min(max_speed, dist)
        vz = (dz/dist) * min(max_speed, dist)
        # add repulsive from others
        rx, ry = 0.0, 0.0
        safe = rospy.get_param('/swarm_simulation/safe_distance', 1.2)
        rep_gain = rospy.get_param('/swarm_simulation/repulsive_gain', 1.0)
        for i,p in self.other_goals.items():
            if i == self.drone_id: continue
            ox = p.x - self.pose['x']
            oy = p.y - self.pose['y']
            d = math.hypot(ox, oy) + 1e-6
            if d < safe:
                # repulsive vector pushing away
                rx -= rep_gain * (safe - d) * (ox / d)
                ry -= rep_gain * (safe - d) * (oy / d)
        cmd = Twist()
        cmd.linear.x = vx + rx
        cmd.linear.y = vy + ry
        cmd.linear.z = vz
        self.pub_cmd.publish(cmd)
        # naive update of local 'pose' for demo visualization (not physical)
        self.pose['x'] += cmd.linear.x * 0.1
        self.pose['y'] += cmd.linear.y * 0.1
        self.pose['z'] += cmd.linear.z * 0.1

if __name__ == '__main__':
    ctrl = FormationController()
    rospy.spin()
