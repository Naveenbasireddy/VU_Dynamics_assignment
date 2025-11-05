#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
import math

drone_id = rospy.get_param("~drone_id", 0)
goal = None

def cb(msg):  global goal; goal = msg

rospy.init_node(f'controller_{drone_id}')
rospy.Subscriber(f"/drone{drone_id}/setpoint", PoseStamped, cb)
pub = rospy.Publisher(f"/drone{drone_id}/cmd_vel", Twist, queue_size=5)
rate = rospy.Rate(10)

pos = [0, 0, 0]
while not rospy.is_shutdown():
    if goal:
        dx, dy, dz = goal.pose.position.x - pos[0], goal.pose.position.y - pos[1], goal.pose.position.z - pos[2]
        vx, vy, vz = 0.4*dx, 0.4*dy, 0.6*dz
        pos[0] += vx*0.1; pos[1] += vy*0.1; pos[2] += vz*0.1
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = vx, vy, vz
        pub.publish(tw)
    rate.sleep()
