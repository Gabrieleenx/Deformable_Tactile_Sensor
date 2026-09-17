#!/usr/bin/env python3
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

def main():
    rospy.init_node('object_to_tactile_direct_broadcaster')

    # TF buffer and listener to read existing transforms
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # TF broadcaster to publish the new transform
    br = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(50)  # 50 Hz

    while not rospy.is_shutdown():
        try:
            # Lookup transform: object_frame -> tactile_sensor
            trans = tf_buffer.lookup_transform(
                'object_frame',
                'tactile_sensor',
                rospy.Time(0),
                rospy.Duration(1.0)  # timeout
            )

            # Create a new TransformStamped message
            t_new = TransformStamped()
            t_new.header.stamp = rospy.Time.now()
            t_new.header.frame_id = 'object_frame'
            t_new.child_frame_id = 'tactile_sensor_direct'
            t_new.transform.translation = trans.transform.translation
            t_new.transform.rotation = trans.transform.rotation

            # Publish the transform
            br.sendTransform(t_new)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "Transform not found yet.")
            pass

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
