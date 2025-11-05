#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

rospy.init_node('takeoff_manager')
pubs = [rospy.Publisher(f"/drone{i}/setpoint", PoseStamped, queue_size=10) for i in range(3)]
rate = rospy.Rate(5)

while not rospy.is_shutdown():
    for p in pubs:
        msg = PoseStamped()
        msg.pose.position.z = 10.0
        msg.pose.orientation.w = 1.0
        p.publish(msg)
    rate.sleep()
