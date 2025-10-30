#!/usr/bin/env python3
import rospy, math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

def load_params():
    num = rospy.get_param('~num_drones', rospy.get_param('/swarm_simulation/num_drones', 4))
    r = rospy.get_param('~formation_radius', rospy.get_param('/swarm_simulation/formation_radius', 4.0))
    return int(num), float(r)

class SwarmManager:
    def __init__(self):
        rospy.init_node('swarm_manager')
        self.num_drones, self.radius = load_params()
        self.pubs = [rospy.Publisher(f'/drone{i}/goal', PoseStamped, queue_size=1) for i in range(self.num_drones)]
        rospy.Subscriber('/target_pose', PoseStamped, self.target_cb)
        rospy.loginfo("Swarm manager started with %d drones", self.num_drones)

    def target_cb(self, msg):
        for i, pub in enumerate(self.pubs):
            angle = 2 * math.pi * i / self.num_drones
            g = PoseStamped()
            g.header.stamp = rospy.Time.now()
            g.header.frame_id = "world"
            g.pose.position.x = msg.pose.position.x + self.radius * math.cos(angle)
            g.pose.position.y = msg.pose.position.y + self.radius * math.sin(angle)
            g.pose.position.z = rospy.get_param('/swarm_simulation/takeoff_alt', 10.0)
            pub.publish(g)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    sm = SwarmManager()
    sm.spin()
