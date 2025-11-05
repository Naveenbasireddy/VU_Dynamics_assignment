#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

target = None
def cb(msg):  global target; target = msg

rospy.init_node('coordinator_node')
rospy.Subscriber('/target/pose_smooth', PoseStamped, cb)
pubs = [rospy.Publisher(f"/drone{i}/setpoint", PoseStamped, queue_size=5) for i in range(3)]
rate = rospy.Rate(10)

offsets = [(0, 0), (2, -2), (-2, -2)]

while not rospy.is_shutdown():
    if target:
        for i, pub in enumerate(pubs):
            msg = PoseStamped()
            msg.header.frame_id = "world"
            msg.pose.position.x = target.pose.position.x + offsets[i][0]
            msg.pose.position.y = target.pose.position.y + offsets[i][1]
            msg.pose.position.z = 10.0
            msg.pose.orientation.w = 1.0
            pub.publish(msg)
    rate.sleep()
