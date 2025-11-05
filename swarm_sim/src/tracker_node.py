#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

latest = None

def cb(msg):
    global latest
    latest = msg

rospy.init_node('tracker_node')
rospy.Subscriber('/target/pose', PoseStamped, cb)
pub = rospy.Publisher('/target/pose_smooth', PoseStamped, queue_size=5)
rate = rospy.Rate(10)

while not rospy.is_shutdown():
    if latest:
        pub.publish(latest)
    rate.sleep()
