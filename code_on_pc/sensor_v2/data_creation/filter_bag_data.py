#!/usr/bin/env python3
import rosbag
import rospy
import tf
import pandas as pd
import bisect
from get_height_map import ThicknessMapper  #
import numpy as np


def process_bag(BAG_FILE,
    mesh1_file,
    mesh2_file,
    offset=0.0,
    resolution=0.0002,
    xy_range=0.02,
    parent_frame="object_frame",
    child_frame="tactile_sensor_direct"):

    # --- Load bag ---
    bag = rosbag.Bag(BAG_FILE)

    # --- Initialize TF transformer ---
    tfm = tf.TransformerROS(True, rospy.Duration(1000.0))  # keep 200s of history

    # --- Load all dynamic TF messages ---
    tf_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/tf']):
        for transform in msg.transforms:
            tf_msgs.append((transform.header.stamp.to_sec(), transform))
    tf_msgs.sort(key=lambda x: x[0])
    tf_times = [ts for ts, _ in tf_msgs]

    for ts, transform in tf_msgs:
        tfm.setTransform(transform)

    # --- Cache FT messages ---
    ft_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/tactile/current_force']):
        ft_msgs.append((t.to_sec(), msg))
    ft_msgs.sort(key=lambda x: x[0])
    ft_times = [ts for ts, _ in ft_msgs]

    # --- Helper: find closest message in time ---
    def find_closest_msg(msg_times, msgs, query_time):
        idx = bisect.bisect_left(msg_times, query_time)
        if idx == 0:
            return msgs[0][1]
        if idx >= len(msg_times):
            return msgs[-1][1]
        before, after = msgs[idx-1][1], msgs[idx][1]
        return before if abs(msg_times[idx-1] - query_time) <= abs(msg_times[idx] - query_time) else after

    # --- Helper: find closest TF timestamp ---
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

    # --- Load thickness mapper ---
    mapper = ThicknessMapper(mesh1_file, mesh2_file, resolution=resolution, xy_range=xy_range)

    # --- Collect data ---
    above_rows = []
    below_rows = []
    start_time = 0.0

    count = 0
    start = None
    end = None

    for i, (topic, tact_msg, t) in enumerate(bag.read_messages(topics=['/tactile_reader/tactile_sensor'])):
        ts = t.to_sec()
        if start is None:
            start = ts
        end = ts
        count += 1
    print("count =", count)
    print("start =", start)
    print("end =", end)
    print("duration =", end - start)

    for i, (topic, tact_msg, t) in enumerate(bag.read_messages(topics=['/tactile_reader/tactile_sensor'])):
        tact_time = t.to_sec()
        if i == 0:
            start_time = tact_time

        # Closest FT message
        ft_msg = find_closest_msg(ft_times, ft_msgs, tact_time)

        # Closest TF timestamp
        closest_tf_time = find_closest_tf_time(tact_time)
        if closest_tf_time is None:
            print("here")
            continue

        try:
            # Lookup transform between parent and child frame
            trans, rot = tfm.lookupTransform(parent_frame, child_frame,
                                             rospy.Time.from_sec(closest_tf_time))
            x, y, z = trans
            qx, qy, qz, qw = rot
        except Exception:
            print("not aviable")
            continue  # skip if transform not available



        # Flatten tactile sensor data
        hall_flat = [v for s in tact_msg.hall_sensor for v in [s.x, s.y, s.z]]
        mouse_flat = [v for s in tact_msg.mouse_sensor for v in [s.x, s.y]]

        # Build row
        row = {
            "time": tact_time,
            "hall_data": hall_flat,
            "mouse_data": mouse_flat,
            "fx": ft_msg.wrench.force.x,
            "fy": ft_msg.wrench.force.y,
            "fz": ft_msg.wrench.force.z,
            "tx": ft_msg.wrench.torque.x,
            "ty": ft_msg.wrench.torque.y,
            "tz": ft_msg.wrench.torque.z,
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw
        }
        # Downsample based on z
        if z > 0.03:
            if i % 800 == 0:
                # Convert TF to 4x4 pose matrix (meters)
                T = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
                T[:3, 3] = [x, y, z]
                
                print(False, ft_msg.wrench.force.z, z, tact_time-start_time)   # prints True or False

                # --- Get thickness map ---
                thickness_map = mapper.get_thickness_map(pose=T, z_offset=offset, show_scene=False)
                row["thickness_map"] = thickness_map
                above_rows.append(row)
        else:
            if i % 50 == 0:
                # Convert TF to 4x4 pose matrix (meters)
                T = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
                T[:3, 3] = [x, y, z]

                # --- Get thickness map ---
                thickness_map = mapper.get_thickness_map(pose=T, z_offset=offset, show_scene=False)

                is_nonzero = np.any(thickness_map)
                print(is_nonzero, ft_msg.wrench.force.z, z, tact_time-start_time, "hi")   # prints True or False
                row["thickness_map"] = thickness_map
                below_rows.append(row)

    bag.close()

    # --- Convert to pandas DataFrames ---
    df_above = pd.DataFrame(above_rows)
    df_below = pd.DataFrame(below_rows)

    return df_above, df_below


if __name__ == "__main__":
    BAG_FILE = ""
    df_above, df_below = process_bag(BAG_FILE)

    print("Above 25mm:", df_above.shape)
    print("Below 25mm:", df_below.shape)
    print(df_above.columns)
    if not df_above.empty:
        print(df_above.iloc[0])
