#!/usr/bin/env python3
import rospy, math
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node('detection_node')
    pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=5)
    rate = rospy.Rate(10)
    start = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - start
        x = 2.5 * math.cos(0.25*t)    # moves in a circle
        y = 2.0 * math.sin(0.25*t)
        # occasionally "occlude" target for 3 seconds every 14 seconds
        visible = (int(t) % 14) < 11
        if visible:
            p = PoseStamped()
            p.header.stamp = rospy.Time.now()
            p.header.frame_id = "world"
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = 0.5
            pub.publish(p)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('detection_node', anonymous=True)
    try:
        main()
    except rospy.ROSInterruptException:
        pass
