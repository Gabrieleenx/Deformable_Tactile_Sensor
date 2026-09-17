import rosbag
import numpy as np
from geometry_msgs.msg import WrenchStamped
import matplotlib as mpl
mpl.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns


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
bag_path = ""             # path to your bag file
topic_A = "/netft_data1"              # high-rate topic
topic_B = "/tactile_mapping/wrench"              # low-rate topic
max_time_diff = 0.005               # seconds allowed between matched samples
N_SAMPLES = 5000                    # desired number of output samples (manual downsampling)

# === READ BAG DATA ===
print("Reading bag...")
bag = rosbag.Bag(bag_path)

msgs_A, msgs_B = [], []
for topic, msg, t in bag.read_messages(topics=[topic_A, topic_B]):
    if topic == topic_A:
        msgs_A.append((msg.header.stamp.to_sec(), msg.wrench.force.z))
    elif topic == topic_B:
        msgs_B.append((msg.header.stamp.to_sec(), msg.wrench.force.z))
bag.close()

print(f"Loaded {len(msgs_A)} messages from {topic_A}, {len(msgs_B)} from {topic_B}")

# === CONVERT TO ARRAYS ===
times_A = np.array([t for t, _ in msgs_A])
fz_A = np.array([f for _, f in msgs_A])
times_B = np.array([t for t, _ in msgs_B])
fz_B = np.array([f for _, f in msgs_B])

# === CREATE UNIFORM DOWNSAMPLED TIMESTAMPS ===
t_min = max(times_A[0], times_B[0])
t_max = min(times_A[-1], times_B[-1])
uniform_times = np.linspace(t_min, t_max, N_SAMPLES)

# === INTERPOLATE BOTH TOPICS TO THESE TIMES ===
# Find nearest or linear interpolation
fz_A_interp = np.interp(uniform_times, times_A, fz_A)
fz_B_interp = np.interp(uniform_times, times_B, fz_B)

print(f"Downsampled to {N_SAMPLES} synchronized samples between {t_min:.3f}–{t_max:.3f} s.")

# === OPTIONAL: OFFSET REMOVE ===
#fz_A_interp -= np.mean(fz_A_interp)
#fz_B_interp -= np.mean(fz_B_interp)
from matplotlib.collections import LineCollection
from matplotlib import cm

# Assume fz_A_interp and fz_B_interp are your arrays
points = np.array([fz_A_interp, fz_B_interp]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a colormap (here using 'viridis')
#lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(0, len(fz_A_interp)))
#lc.set_array(np.arange(len(fz_A_interp)))  # color changes with index
# Elapsed time (s)
time = uniform_times - uniform_times[0]

lc = LineCollection(
    segments,
    cmap='coolwarm',          # or 'plasma'
    norm=plt.Normalize(time.min(), time.max())
)
lc.set_array(time[:-1])      # one value per segment

lc.set_linewidth(2)
lc.set_alpha(0.3)

fig, ax = plt.subplots(figsize=(4, 3))
ax.add_collection(lc)
ax.set_xlim(fz_A_interp.min(), fz_A_interp.max())
ax.set_ylim(fz_B_interp.min(), fz_B_interp.max())

ax.set_xlabel('$f_z$ from F/T sensor $[N]$')
ax.set_ylabel('$f_z$ from tactile sensor $[N]$')
ax.set_title('Hysteresis Plot 50 cycles')
ax.grid(True)
ax.axis('equal')

cbar = fig.colorbar(lc, ax=ax)
cbar.set_label('Elapsed time (s)')

plt.tight_layout()
plt.savefig("hysterisis_1.pdf")
plt.close()


# === PLOT HYSTERESIS ===
plt.figure(figsize=(4, 4))
plt.plot(fz_A_interp, fz_B_interp, '-o', markersize=2)

plt.xlabel(f'$f_z$ from F/T sensor $[N]$')
plt.ylabel(f'$f_z$ from tactile sensor $[N]$')
plt.title(f'Hysteresis Plot 50 cycles)')
plt.grid(True)
plt.axis('equal')


plt.tight_layout()
plt.savefig("hysterisis_2.pdf")
plt.close()
