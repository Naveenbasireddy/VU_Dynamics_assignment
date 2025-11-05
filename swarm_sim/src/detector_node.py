#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

rospy.init_node('detector_node')
pub = rospy.Publisher('/target/pose', PoseStamped, queue_size=5)
rate = rospy.Rate(10)
t = 0.0

while not rospy.is_shutdown():
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x = 5 + 2 * np.cos(t)
    msg.pose.position.y = 2 * np.sin(t)
    msg.pose.position.z = 0.0
    pub.publish(msg)
    t += 0.1
    rate.sleep()
