#!/usr/bin/env python3
import rosbag
import rospy
import tf
import numpy as np
import bisect
import math

import rosbag
import rospy
import tf
import math

import rosbag
import rospy
import tf
import math


def rotate_subsensor_to_sensor(dx, dy, theta):
    # rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # if dx and dy are single values, make them arrays
    vec = np.vstack([dx, dy])  # shape: 2 x N (or 2 x 1 if single)

    rotated = R @ vec

    return rotated[0], rotated[1]


def dxy_to_mm_s(dx, dy, hz, cpmm, theta):
    dx_, dy_ = rotate_subsensor_to_sensor(dx, dy, theta)
    vx = hz*dx_/cpmm
    vy = hz*dy_/cpmm
    return vx, vy

def compute_xy_distance(bag_file, parent_frame="object_frame", child_frame="tactile_sensor_direct"):
    # Load bag
    bag = rosbag.Bag(bag_file)

    # Initialize TF transformer
    tfm = tf.TransformerROS(True, rospy.Duration(1000.0))

    # --- Collect only relevant TFs ---
    tf_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/tf']):
        for transform in msg.transforms:
            if (transform.header.frame_id == parent_frame and
                transform.child_frame_id == child_frame):
                tf_msgs.append((transform.header.stamp.to_sec(), transform))
    bag.close()

    if not tf_msgs:
        print("No TF data found in bag.")
        return None

    # Sort and load into transformer
    tf_msgs.sort(key=lambda x: x[0])
    for _, transform in tf_msgs:
        tfm.setTransform(transform)

    # --- Find first available transform ---
    first_pose, first_rot, first_time = None, None, None
    for ts, _ in tf_msgs:
        try:
            trans, rot = tfm.lookupTransform(parent_frame, child_frame, rospy.Time.from_sec(ts))
            rot_inv = tf.transformations.quaternion_conjugate(rot)
            trans_inv = tf.transformations.quaternion_matrix(rot_inv)[:3, :3].dot(-np.array(trans))
            #first_pose, first_rot, first_time = trans, rot, ts
            first_pose, first_rot, first_time = trans_inv, rot_inv, ts
            break
        except Exception:
            continue

    # --- Find last available transform ---
    last_pose, last_rot, last_time = None, None, None
    for ts, _ in reversed(tf_msgs):
        try:
            trans, rot = tfm.lookupTransform(parent_frame, child_frame, rospy.Time.from_sec(ts))
            rot_inv = tf.transformations.quaternion_conjugate(rot)
            trans_inv = tf.transformations.quaternion_matrix(rot_inv)[:3, :3].dot(-np.array(trans))
            #last_pose, last_rot, last_time = trans, rot, ts
            last_pose, last_rot, last_time = trans_inv, rot_inv, ts
            break
        except Exception:
            continue
    
    

    if first_pose is None or last_pose is None:
        print("Could not compute distance.")
        return None

    # --- Compute XY distance ---
    dx = last_pose[0] - first_pose[0]
    dy = last_pose[1] - first_pose[1]
    distance_xy = math.sqrt(dx**2 + dy**2)

    # --- Compute yaw difference ---
    yaw0 = tf.transformations.euler_from_quaternion(first_rot)[2]
    yaw1 = tf.transformations.euler_from_quaternion(last_rot)[2]
    dyaw = yaw1 - yaw0 
    return {
        "distance_xy": distance_xy,
        "first_pose": first_pose,
        "first_time": first_time,
        "last_pose": last_pose,
        "last_time": last_time,
        "dx": dx,
        "dy": dy,
        "dyaw": dyaw  # <--- rotation around z (radians)
    }




def estimate_cpmm(bag_file_name):
    #dist = compute_xy_distance(bag_file_name, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    dist = compute_xy_distance(bag_file_name, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    
    bag = rosbag.Bag(bag_file_name)
    dx1_sum = 0
    dy1_sum = 0
    dx2_sum = 0
    dy2_sum = 0
    for i, (topic, tact_msg, t) in enumerate(bag.read_messages(topics=['/tactile_reader/tactile_sensor'])):
        tact_time = t.to_sec()
        if tact_time < dist["first_time"]:
            continue
        dx1_sum += tact_msg.mouse_sensor[0].x
        dy1_sum += tact_msg.mouse_sensor[0].y
        dx2_sum += tact_msg.mouse_sensor[1].x
        dy2_sum += tact_msg.mouse_sensor[1].y

    bag.close()


    # rotation angle

    theta_1 = np.arctan2(dist["dy"], dist["dx"]) - np.arctan2(dy1_sum, dx1_sum)
    theta_2 = np.arctan2(dist["dy"], dist["dx"]) - np.arctan2(dy2_sum, dx2_sum)


    counts_1 = np.linalg.norm([dx1_sum, dy1_sum])
    counts_2 = np.linalg.norm([dx2_sum, dy2_sum])
    xy_mm = dist["distance_xy"]*1e3
    return counts_1/xy_mm, counts_2/xy_mm, theta_1, theta_2





def estimate_pose(bag_file_name, theta_1, theta_2, cpmm_1, cpmm_2, hz):
    dist = compute_xy_distance(bag_file_name, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    dyaw = dist["dyaw"] # rad
    bag = rosbag.Bag(bag_file_name)
    dx1_dist = 0
    dy1_dist = 0
    dx2_dist = 0
    dy2_dist = 0

    dist_1x = 0
    dist_1y = 0
    dist_2x = 0
    dist_2y = 0

    for i, (topic, tact_msg, t) in enumerate(bag.read_messages(topics=['/tactile_reader/tactile_sensor'])):
        tact_time = t.to_sec()
        if tact_time < dist["first_time"]:
            continue

        vx_1, vy_1 = dxy_to_mm_s(tact_msg.mouse_sensor[0].x, tact_msg.mouse_sensor[0].y, hz, cpmm_1, -theta_1) #mm/s
        vx_2, vy_2 = dxy_to_mm_s(tact_msg.mouse_sensor[1].x, tact_msg.mouse_sensor[1].y, hz, cpmm_2, -theta_2)


        dist_1x += tact_msg.mouse_sensor[0].x
        dist_1y += tact_msg.mouse_sensor[0].y
        dist_2x += tact_msg.mouse_sensor[1].x
        dist_2y += tact_msg.mouse_sensor[1].y
        

        dx1_dist += vx_1*(1/hz) # mm
        dy1_dist += vy_1*(1/hz)
        dx2_dist += vx_2*(1/hz)
        dy2_dist += vy_2*(1/hz)

    bag.close()
    

    def get_sensor_pose(dx_sub, dy_sub):
        dist = np.linalg.norm([dx_sub, dy_sub])
        r = dist/abs(dyaw)
        theta = np.arctan2(dy_sub, dx_sub) + np.pi/2
        dx = r*np.cos(theta)
        dy = r*np.sin(theta)
        return dx, dy 


    x1, y1 = get_sensor_pose(dx1_dist, dy1_dist)
    x2, y2 = get_sensor_pose(dx2_dist, dy2_dist)


    return x1, y1, x2, y2
    

    

if __name__ == "__main__":
    #sensor_data, mouse_data = generate_synthetic_data(N=500, x=-0.001, y=0.01, angle=0.2)

    bag_file = "code_on_pc/sensor_v2/estimate_sensor_pose/data/cpi_test.bag"
    bag_file_pose = "code_on_pc/sensor_v2/estimate_sensor_pose/data/rotation_test.bag"
    HZ = 250
    cpmm_1, cpmm_2, theta_1, theta_2 = estimate_cpmm(bag_file)
    x1, y1, x2, y2 = estimate_pose(bag_file_pose, theta_1, theta_2, cpmm_1, cpmm_2, HZ)

    print("cpmm_1", cpmm_1, "theta_1", theta_1, "x1", x1, "y1", y1)
    print("cpmm_2", cpmm_2, "theta_2", theta_2, "x2", x2, "y2", y2)
