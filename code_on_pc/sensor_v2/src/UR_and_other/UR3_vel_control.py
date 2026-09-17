#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TwistStamped

def main():
    rospy.init_node('ee_velocity_controller')

    pub = rospy.Publisher(
        '/scaled_cartesian_velocity_controller/command',
        TwistStamped,
        queue_size=1
    )

    rate = rospy.Rate(50)  # 50 Hz

    while not rospy.is_shutdown():
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base"  # Or your chosen reference frame

        # Linear velocity in m/s
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.05  # move up

        # Angular velocity in rad/s
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass