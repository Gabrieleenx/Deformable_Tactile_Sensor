#!/usr/bin/env python3
import rosbag
import rospy
import tf
import bisect
import numpy as np

def find_contact_z(bag_file, parent_frame="object_frame", child_frame="tactile_sensor_direct", force_threshold=0.20):
    """
    Extracts z-position of object_frame -> tactile_sensor_direct when contact occurs.

    Args:
        bag_file (str): Path to the rosbag file
        parent_frame (str): Reference frame (default "object_frame")
        child_frame (str): Sensor frame (default "tactile_sensor_direct")
        force_threshold (float): Threshold on normal force (N) for detecting contact

    Returns:
        float: z-position at contact (in meters), or None if not found
    """

    bag = rosbag.Bag(bag_file)
    print("huh")
    tfm = tf.TransformerROS(True, rospy.Duration(200.0))

    # --- Load static transforms ---
    for topic, msg, t in bag.read_messages(topics=['/tf_static']):
        for transform in msg.transforms:
            transform.header.stamp = rospy.Time(0)
            tfm.setTransform(transform)
    # --- Load dynamic transforms ---
    tf_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/tf']):
        for transform in msg.transforms:
            tf_msgs.append((transform.header.stamp.to_sec(), transform))
    tf_msgs.sort(key=lambda x: x[0])
    tf_times = [ts for ts, _ in tf_msgs]

    for ts, transform in tf_msgs:
        tfm.setTransform(transform)

    # --- Load FT data ---
    ft_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/netft_data1']):
        ft_msgs.append((t.to_sec(), msg))
    ft_msgs.sort(key=lambda x: x[0])
    ft_times = [ts for ts, _ in ft_msgs]
    # --- Helper: find closest TF time ---
    def find_closest_tf_time(query_time):
        if not tf_times:
            return None
        idx = bisect.bisect_left(tf_times, query_time)
        if idx == 0:
            return tf_times[0]
        if idx >= len(tf_times):
            return tf_times[-1]
        before, after = tf_times[idx-1], tf_times[idx]
        return before if abs(before - query_time) <= abs(after - query_time) else after

    # --- Find contact ---
    z_at_contact = None

    for t_sec, ft_msg in ft_msgs:
        normal_force = abs(ft_msg.wrench.force.z)  # assume z is contact normal
        
        if normal_force > force_threshold:  # contact detected
            print("normal_force", normal_force)
            closest_tf_time = find_closest_tf_time(t_sec)
            if closest_tf_time is None:
                continue
            try:
                trans, rot = tfm.lookupTransform(parent_frame, child_frame,
                                                 rospy.Time.from_sec(closest_tf_time))
                z_at_contact = trans[2]
                break  # first contact → stop
            except Exception:
                continue

    bag.close()
    return z_at_contact


if __name__ == "__main__":
    BAG_FILE = ""
    z_contact = find_contact_z(BAG_FILE, force_threshold=0.25)
    if z_contact is not None:
        print(f"Contact detected at z = {z_contact:.4f} m")
    else:
        print("No contact detected in this bag.")
