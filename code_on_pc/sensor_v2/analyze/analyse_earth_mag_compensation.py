#!/usr/bin/env python3
import rosbag
import numpy as np
import rospy, tf
from geometry_msgs.msg import WrenchStamped
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")   # headless-safe backend
from scipy.spatial.transform import Rotation as R

# === STYLE SETTINGS ===
mpl.rcParams['pdf.fonttype'] = 42     # Embed TrueType fonts
mpl.rcParams['ps.fonttype']  = 42

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['Times New Roman']

mpl.rcParams['mathtext.fontset'] = 'dejavuserif'  # NOT 'cm'

mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 1.6})

# === CONFIGURATION ===
bag_uncomp = ""  # with compenstaion off
bag_comp   = ""  # with compenstaion on

topic_tactile = "/tactile_mapping/wrench"
parent_frame = "object_frame"
child_frame  = "tactile_sensor_direct"

# === FUNCTIONS ===
def read_wrench_data(bag_path, topic):
    """Read wrench messages (force, torque) and timestamps."""
    print(f"Reading {bag_path} ...")
    bag = rosbag.Bag(bag_path)
    times, forces, torques = [], [], []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        t = msg.header.stamp.to_sec()
        f = msg.wrench.force
        tau = msg.wrench.torque
        times.append(t)
        forces.append([f.x, f.y, f.z])
        torques.append([tau.x, tau.y, tau.z])
    bag.close()
    return np.array(times), np.array(forces), np.array(torques)


def read_tf_quats(bag_path, parent, child):
    """Extract quaternions from TF transforms between specific frames."""
    print(f"Extracting TFs from {bag_path} ...")
    bag = rosbag.Bag(bag_path)
    tfm = tf.TransformerROS(True, rospy.Duration(200.0))

    # Load static transforms
    for _, msg, _ in bag.read_messages(topics=['/tf_static']):
        for t in msg.transforms:
            t.header.stamp = rospy.Time(0)
            tfm.setTransform(t)

    # Collect transforms between given frames
    tf_msgs = []
    for _, msg, _ in bag.read_messages(topics=['/tf']):
        for t in msg.transforms:
            if t.header.frame_id == parent and t.child_frame_id == child:
                tf_msgs.append((t.header.stamp.to_sec(), t.transform.rotation))
    bag.close()

    if not tf_msgs:
        raise RuntimeError(f"No TFs found between {parent} → {child} in {bag_path}")

    tf_msgs.sort(key=lambda x: x[0])
    unique_times, indices = np.unique([t for t, _ in tf_msgs], return_index=True)
    quats = np.array([
        [tf_msgs[i][1].x, tf_msgs[i][1].y, tf_msgs[i][1].z, tf_msgs[i][1].w]
        for i in indices
    ])

    # Fix quaternion sign flips for continuity
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] = -quats[i]

    return unique_times, quats


def match_tf_to_wrench(wrench_times, tf_times, tf_quats):
    """Match each wrench timestamp to the nearest quaternion and get Euler angles."""
    tf_times = np.array(tf_times)
    indices = np.searchsorted(tf_times, wrench_times)
    indices = np.clip(indices, 1, len(tf_times) - 1)
    left = tf_times[indices - 1]
    right = tf_times[indices]
    choose_right = np.abs(wrench_times - left) > np.abs(wrench_times - right)
    matched_indices = indices.copy()
    matched_indices[choose_right] = indices[choose_right]
    matched_quats = tf_quats[matched_indices]

    # Convert to Euler angles (in degrees)
    eulers = R.from_quat(matched_quats).as_euler('xyz', degrees=True)

    # Unwrap angles to remove ±180° jumps
    eulers = np.unwrap(np.deg2rad(eulers), axis=0)
    eulers = np.rad2deg(eulers)

    return eulers


# === MAIN ===
times_U, forces_U, torques_U = read_wrench_data(bag_uncomp, topic_tactile)
times_C, forces_C, torques_C = read_wrench_data(bag_comp, topic_tactile)
tf_times_U, quats_U = read_tf_quats(bag_uncomp, parent_frame, child_frame)
tf_times_C, quats_C = read_tf_quats(bag_comp, parent_frame, child_frame)

# --- Align start times ---
time_shift = times_U[0] - times_C[0]
times_C_shifted = times_C + time_shift
tf_times_C_shifted = tf_times_C + time_shift

# --- Match TFs to wrench timestamps ---
angles_U = match_tf_to_wrench(times_U, tf_times_U, quats_U)
angles_C = match_tf_to_wrench(times_C_shifted, tf_times_C_shifted, quats_C)

# --- Relative time for plotting ---
t_rel_U = times_U - times_U[0]
t_rel_C = times_C_shifted - times_U[0]

# === PLOT ===
fig, axs = plt.subplots(3, 1, figsize=(5, 4), sharex=True)
colors = ['tab:blue', 'tab:orange', 'tab:green']
labels_force = ['Fx', 'Fy', 'Fz']
labels_ang = [r'Roll [°]', r'Pitch [°]', r'Yaw [°]']

# Optional: downsample for plotting clarity
PLOT_DOWNSAMPLE = max(1, len(t_rel_U) // 500)  # limit to ~3000 visible points
t_rel_U_ds = t_rel_U[::PLOT_DOWNSAMPLE]
t_rel_C_ds = t_rel_C[::PLOT_DOWNSAMPLE]
forces_U_ds = forces_U[::PLOT_DOWNSAMPLE]
forces_C_ds = forces_C[::PLOT_DOWNSAMPLE]
torques_U_ds = torques_U[::PLOT_DOWNSAMPLE]
torques_C_ds = torques_C[::PLOT_DOWNSAMPLE]
angles_U_ds = angles_U[::PLOT_DOWNSAMPLE]


# --- Forces ---
for i in range(3):
    axs[0].plot(t_rel_U_ds, forces_U_ds[:, i], '--', color=colors[i], alpha=0.8, label=f'Uncomp {labels_force[i]}')
    axs[0].plot(t_rel_C_ds, forces_C_ds[:, i], '-', color=colors[i], alpha=0.8, label=f'Comp {labels_force[i]}')
axs[0].set_ylabel("Force [N]")
axs[0].set_title("Tactile Sensor Forces")
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(frameon=False, ncol=3)

# --- Torques ---
for i in range(3):
    axs[1].plot(t_rel_U_ds, torques_U_ds[:, i], '--', alpha=0.8, color=colors[i])
    axs[1].plot(t_rel_C_ds, torques_C_ds[:, i], '-', alpha=0.8, color=colors[i])
axs[1].set_ylabel("Torque [Nm]")
axs[1].set_title("Tactile Sensor Torques")
axs[1].grid(True, linestyle='--', alpha=0.6)

# --- Angles ---
for i in range(3):
    axs[2].plot(t_rel_U_ds, angles_U_ds[:, i], '--', color=colors[i], label=f'Uncomp {labels_ang[i]}')
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Angle [°]")
axs[2].set_title("Sensor Orientation (Roll, Pitch, Yaw)")
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(frameon=False, ncol=3)

plt.tight_layout()
plt.savefig("external_compensation.pdf")
plt.close()

