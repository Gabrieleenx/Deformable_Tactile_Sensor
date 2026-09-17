import rosbag
import csv
import numpy as np
from scipy.optimize import curve_fit

bag = rosbag.Bag(
    "/sensor_inteferance.bag" # bag where gripper opens and closes
)

# Hall sensors to fit
hall_indices = range(13)

CONTACT_POS = 6.5


def exp_decay(x, A, k, C):
    return A * np.exp(-k * (x - CONTACT_POS)) + C


# ---------------------------
# Storage
# ---------------------------
t1, t2, tm = [], [], []
position = []

hall1 = {i: {"x": [], "y": [], "z": []} for i in hall_indices}
hall2 = {i: {"x": [], "y": [], "z": []} for i in hall_indices}

t0 = None

# ---------------------------
# Read bag
# ---------------------------
for topic, msg, t in bag.read_messages():

    if t0 is None:
        t0 = t.to_sec()

    ts = t.to_sec() - t0

    if topic == "/tactile_reader/tactile_sensor1":

        t1.append(ts)

        for i in hall_indices:
            hs = msg.hall_sensor[i]
            hall1[i]["x"].append(hs.x)
            hall1[i]["y"].append(hs.y)
            hall1[i]["z"].append(hs.z)

    elif topic == "/tactile_reader/tactile_sensor2":

        t2.append(ts)

        for i in hall_indices:
            hs = msg.hall_sensor[i]
            hall2[i]["x"].append(hs.x)
            hall2[i]["y"].append(hs.y)
            hall2[i]["z"].append(hs.z)

    elif topic == "/gripper_monitor":

        tm.append(ts)
        position.append(msg.position)

bag.close()

# ---------------------------
# Convert to numpy
# ---------------------------
t1 = np.array(t1)
t2 = np.array(t2)
tm = np.array(tm)

position = np.array(position)

for i in hall_indices:
    for axis in ["x", "y", "z"]:
        hall1[i][axis] = np.array(hall1[i][axis])
        hall2[i][axis] = np.array(hall2[i][axis])

# ---------------------------
# Interpolate gripper position
# ---------------------------
pos1 = np.interp(t1, tm, position)
pos2 = np.interp(t2, tm, position)

# ---------------------------
# Time window
# ---------------------------
t_start = 8.2
t_end = 11.2

mask1 = (
    (t1 >= t_start)
    & (t1 <= t_end)
    & (pos1 >= CONTACT_POS)
)

mask2 = (
    (t2 >= t_start)
    & (t2 <= t_end)
    & (pos2 >= CONTACT_POS)
)

# ---------------------------
# Fit all sensors
# ---------------------------
fit_results_s1 = []
fit_results_s2 = []

for i in hall_indices:

    for axis in ["x", "y", "z"]:

        # ==========================
        # Sensor 1
        # ==========================
        x = pos1[mask1]
        y = hall1[i][axis][mask1]

        try:

            p0 = [
                y[0] - y[-1],
                0.08,
                y[-1],
            ]

            popt, _ = curve_fit(
                exp_decay,
                x,
                y,
                p0=p0,
                maxfev=10000,
            )

            fit_results_s1.append([
                i,
                axis,
                popt[0],
                popt[1],
                popt[2],
            ])

            print(
                f"S1[{i}] {axis}: "
                f"A={popt[0]:.3f}, "
                f"k={popt[1]:.3f}, "
                f"C={popt[2]:.3f}"
            )

        except Exception:

            fit_results_s1.append([
                i,
                axis,
                np.nan,
                np.nan,
                np.nan,
            ])

            print(f"Fit failed for S1[{i}] {axis}")

        # ==========================
        # Sensor 2
        # ==========================
        x = pos2[mask2]
        y = hall2[i][axis][mask2]

        try:

            p0 = [
                y[0] - y[-1],
                0.08,
                y[-1],
            ]

            popt, _ = curve_fit(
                exp_decay,
                x,
                y,
                p0=p0,
                maxfev=10000,
            )

            fit_results_s2.append([
                i,
                axis,
                popt[0],
                popt[1],
                popt[2],
            ])

            print(
                f"S2[{i}] {axis}: "
                f"A={popt[0]:.3f}, "
                f"k={popt[1]:.3f}, "
                f"C={popt[2]:.3f}"
            )

        except Exception:

            fit_results_s2.append([
                i,
                axis,
                np.nan,
                np.nan,
                np.nan,
            ])

            print(f"Fit failed for S2[{i}] {axis}")

# ---------------------------
# Save CSV files
# ---------------------------
header = ["hall_sensor", "axis", "A", "k", "C"]

with open("sensor1_fit_parameters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(fit_results_s1)

with open("sensor2_fit_parameters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(fit_results_s2)

print("\nSaved:")
print("  sensor1_fit_parameters.csv")
print("  sensor2_fit_parameters.csv")