#!/usr/bin/env python3
import rosbag
import rospy
import tf
import numpy as np
import bisect
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


def sensor_xy_distance(bag_file, Hz=250):
    # Load bag
    bag = rosbag.Bag(bag_file)
    dx = 0
    dy = 0
    dyaw = 0
    for i, (topic, tact_msg, t) in enumerate(bag.read_messages(topics=['/tactile_mapping/velocity'])):
        time_s = tact_msg.header.stamp.to_sec()
        if i == 0:
            dt = 1/Hz
        else:
            dt = time_s - last_time
        last_time = time_s

        dx += tact_msg.vx * dt
        dy += tact_msg.vy * dt
        dyaw += tact_msg.omega * dt
   
    bag.close()

    return {
        "dx": dx,
        "dy": dy,
        "dyaw": dyaw  # <--- rotation around z (radians)
    }


    
def calc_tracing_error(bag_file_x, bag_file_y, bag_file_yaw):
    dist = compute_xy_distance(bag_file_x, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    dist_sensor = sensor_xy_distance(bag_file_x, Hz=250)
    error_pecent_x = (dist_sensor["dx"]/1000)/dist["dx"] - 1 

    dist = compute_xy_distance(bag_file_y, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    dist_sensor = sensor_xy_distance(bag_file_y, Hz=250)
    error_pecent_y = (dist_sensor["dy"]/1000)/dist["dy"] - 1 

    dist = compute_xy_distance(bag_file_yaw, parent_frame="object_frame", child_frame="tactile_sensor_direct")
    dist_sensor = sensor_xy_distance(bag_file_yaw, Hz=250)
    error_pecent_yaw = (dist_sensor["dyaw"])/dist["dyaw"] - 1 

    return error_pecent_x, error_pecent_y, error_pecent_yaw


def print_latex_table(results, names, normal_force_list):
    # Header
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\scriptsize")
    col_format = "l" + "c" * len(names)
    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\hline")

    header = [""] + [n.replace("_", "\\_") for n in names]
    print(" & ".join(header) + " \\\\")
    print("\\hline")

    for fn in normal_force_list:
        # ex row
        row_ex = [f"{fn}N $e_x$"]
        for name in names:
            ex, ey, e_yaw = results[name][fn]
            row_ex.append(f"{ex:.4f}")
        print(" & ".join(row_ex) + " \\\\")

        # ey row
        row_ey = [f"{fn}N $e_y$"]
        for name in names:
            ex, ey, e_yaw = results[name][fn]
            row_ey.append(f"{ey:.4f}")
        print(" & ".join(row_ey) + " \\\\")

        print("\\hline")

    print("\\end{tabular}")
    print("\\caption{Tracing error for different materials and normal forces}")
    print("\\label{tab:tracing_error}")
    print("\\end{table}")

def print_latex_table_yaw(results, names, normal_force_list):
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\scriptsize")
    col_format = "l" + "c" * len(names)
    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\hline")

    header = [""] + [n.replace("_", "\\_") for n in names]
    print(" & ".join(header) + " \\\\")
    print("\\hline")

    for fn in normal_force_list:
        row = [f"{fn}N $e_\\psi$"]
        for name in names:
            ex, ey, e_yaw = results[name][fn]
            row.append(f"{e_yaw:.4f}")
        print(" & ".join(row) + " \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print("\\caption{Yaw tracing error for different materials and normal forces}")
    print("\\label{tab:tracing_error_yaw}")
    print("\\end{table}")


if __name__ == "__main__":
    #sensor_data, mouse_data = generate_synthetic_data(N=500, x=-0.001, y=0.01, angle=0.2)
    dirr = "" # path to folder with bag files

    names = ["plastic_flat", "wood_flat", "paper_flat", "transparent_flat", "plasic_curved_r200", "plasic_curved_r100", "plasic_curved_r50", "plasic_curved_r25"]

    normal_force_list = [1, 2, 4, 8]
    # results[name][fn] = (ex, ey)
    results = {name: {} for name in names}
    for name in names:
        print(name)
        for fn in normal_force_list:
            bag_file_x = dirr + name + "_" + str(fn)  + "N_" + "0_lin_vel.bag"
            bag_file_y = dirr + name + "_" + str(fn)  + "N_" + "90_lin_vel.bag"
            bag_file_yaw = dirr + name + "_" + str(fn) + "N_" + "_rot_vel.bag"
            ex, ey, e_yaw = calc_tracing_error(bag_file_x, bag_file_y, bag_file_yaw)
            results[name][fn] = (ex, ey, e_yaw)
    print_latex_table(results, names, normal_force_list)    
    print_latex_table_yaw(results, names, normal_force_list)


