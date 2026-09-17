#!/usr/bin/env python3
import rosbag
import numpy as np
from geometry_msgs.msg import WrenchStamped
import rospy, tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")   # headless-safe backend
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# --- Plot style settings ---
mpl.rcParams['pdf.fonttype'] = 42     # Embed TrueType fonts
mpl.rcParams['ps.fonttype']  = 42

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['Times New Roman']

mpl.rcParams['mathtext.fontset'] = 'dejavuserif'  # NOT 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.3, rc={"lines.linewidth": 2})

# === CONFIGURATION ===
bag_path = ""
topic_A = "/netft_data1"                # F/T sensor
topic_B = "/tactile_mapping/wrench"     # Tactile sensor
parent_frame = "object_frame"           # reference frame
child_frame = "tactile_sensor_direct"   # sensor frame
N_SAMPLES = 5000                        # number of samples for interpolation

# === READ BAG DATA ===
print("Reading bag...")
bag = rosbag.Bag(bag_path)
msgs_A, msgs_B = [], []

for topic, msg, t in bag.read_messages(topics=[topic_A, topic_B]):
    if topic == topic_A:
        msgs_A.append((msg.header.stamp.to_sec(), msg.wrench.force.z))
    elif topic == topic_B:
        msgs_B.append((msg.header.stamp.to_sec(), msg.wrench.force.z))

print(f"Loaded {len(msgs_A)} messages from {topic_A}, {len(msgs_B)} from {topic_B}")

# Convert to arrays
times_A = np.array([t for t, _ in msgs_A])
fz_A = np.array([f for _, f in msgs_A])
times_B = np.array([t for t, _ in msgs_B])
fz_B = np.array([f for _, f in msgs_B])

# Downsample and synchronize
t_min = max(times_A[0], times_B[0])
t_max = min(times_A[-1], times_B[-1])
uniform_times = np.linspace(t_min, t_max, N_SAMPLES)
fz_A_interp = np.interp(uniform_times, times_A, fz_A)
fz_B_interp = np.interp(uniform_times, times_B, fz_B)
print(f"Downsampled to {N_SAMPLES} synchronized samples between {t_min:.3f}–{t_max:.3f} s.")

# === EXTRACT Z-POSITION FROM TF ===
print("Extracting transforms...")
tfm = tf.TransformerROS(True, rospy.Duration(200.0))

# Load static transforms
for topic, msg, t in bag.read_messages(topics=['/tf_static']):
    for transform in msg.transforms:
        transform.header.stamp = rospy.Time(0)
        tfm.setTransform(transform)

# Load dynamic transforms
tf_msgs = []
for topic, msg, t in bag.read_messages(topics=['/tf']):
    for transform in msg.transforms:
        tf_msgs.append((transform.header.stamp.to_sec(), transform))
tf_msgs.sort(key=lambda x: x[0])
tf_times = [ts for ts, _ in tf_msgs]

# Compute z-positions for uniform_times

z_positions = []
for ts in uniform_times:
    print(ts)
    idx = np.searchsorted(tf_times, ts)
    if idx == 0 or idx >= len(tf_msgs):
        z_positions.append(np.nan)
        continue
    tfm.setTransform(tf_msgs[idx][1])
    try:
        (trans, rot) = tfm.lookupTransform(parent_frame, child_frame, rospy.Time(0))
        z_positions.append(trans[2])
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        z_positions.append(np.nan)

bag.close()


z_positions = np.array(z_positions)
start_pos =  np.nanmax(z_positions)
print('start_pos', start_pos)
z_positions -= start_pos
t_rel = uniform_times - uniform_times[0]

# === PLOT RESULTS ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True,
                               gridspec_kw={'height_ratios': [1.5, 1]})

# --- Force plot ---
ax1.plot(t_rel, fz_A_interp, label='F/T sensor', color='tab:blue', linewidth=1.5)
ax1.plot(t_rel, fz_B_interp, label='Tactile sensor', color='tab:orange', linewidth=1.5)
ax1.set_ylabel('$f_z$ $[N]$')
ax1.set_title('Estimated force drift vs Time')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=False)

# --- Z-position plot ---
# --- Z-position plot with zoom inset ---
ax2.plot(t_rel, z_positions, color='tab:green', linewidth=1.5)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('$z$ $[m]$')
ax2.grid(True, linestyle='--', alpha=0.6)

# --- Create inset zoom ---
x1, x2 = 7.1, 250         # time window (s)
y1, y2 = 0.0031 - start_pos, 0.0035 - start_pos  # z window (m)

axins = inset_axes(ax2, width="35%", height="45%", loc='upper left', borderpad=2)
axins.plot(t_rel, z_positions, color='tab:green', linewidth=1.2)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([0.0032 - np.round(start_pos,4), 0.0035 - np.round(start_pos,4)])
axins.yaxis.tick_right()
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(axis='y', labelsize=7)
axins.yaxis.get_offset_text().set_fontsize(7)
# Draw connectors between inset and main plot
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)
plt.tight_layout()
plt.savefig("drift_4.pdf")
plt.close()
