#!/usr/bin/env python3
import height_calibration
import filter_bag_data
import pandas as pd
import sys

path_to_main_folder = ""
folder_path = path_to_main_folder + "/test_unseen/"
BAG_FILE = folder_path + "calibrate_height.bag"
z_contact = height_calibration.find_contact_z(BAG_FILE, force_threshold=0.3)
if z_contact is not None:
    print(f"Contact detected at z = {z_contact:.4f} m")
else:
    print("No contact detected in this bag.")
print(z_contact)
if z_contact > 0.03:
    sys.exit("Exiting: z_contact is greater than 0.03, adjust force_threshold")    

offset =  z_contact - 0.041
print(offset)



file_names = ["torque_and_xy.bag",
              "line_4mm_rot.bag",
              "square.bag",
              "cylinder_flat_45.bag",
              "100mm_cylinder_curve.bag",
              "25mm_cylinder_curve.bag",
              "cylinder_flat_25.bag",
              "peg_curve.bag",
              "line_1mm.bag",
              "dome_50.bag",
              "peg_flat.bag",
              "dome_100.bag",
              "line_4mm.bag"]



file_names = ["unseen_object"]


# Define mesh file paths
mesh1_file = path_to_main_folder + "/data_creation/mesh_data/sensor_playground v3.stl"
mesh2_file = path_to_main_folder + "/data_creation/mesh_data/sensor.stl"

df_combined = pd.DataFrame()

for name_ in file_names:
    print(" ")
    print(name_)
    print(" ")
    BAG_FILE = folder_path + name_
    df_above, df_below = filter_bag_data.process_bag(BAG_FILE, mesh1_file, mesh2_file, offset=offset)

    df_combined = pd.concat([df_combined, df_above, df_below], ignore_index=True)
df_combined.to_csv(folder_path+"data.csv", index=False)
