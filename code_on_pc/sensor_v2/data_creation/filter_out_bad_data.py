import pandas as pd
import ast
import numpy as np

print("filtering out bad data")
# Load CSV
map_path = "" # path to folder with bag files
df = pd.read_csv(map_path + "data.csv")

def all_zero_matrix(map_str):
    """Return True if all elements in thickness_map matrix are zero."""
    try:
        matrix = np.array(ast.literal_eval(map_str))
        return np.allclose(matrix, 0)
    except Exception:
        return False

# Masks
all_zero_mask = df["thickness_map"].apply(all_zero_matrix)
high_fz_mask = df["fz"].abs() > 0.1

# Combined mask for bad rows
bad_rows_mask = all_zero_mask & high_fz_mask

# Count how many will be removed
num_removed = bad_rows_mask.sum()

# Filter them out
df_filtered = df[~bad_rows_mask]

# Save the cleaned data
output_path = map_path + "data_filtered_2.csv"
df_filtered.to_csv(output_path, index=False)

# Print summary
print(f"Total rows before filtering: {len(df)}")
print(f"Rows removed (all-zero thickness_map & |fz| > 0.1): {num_removed}")
print(f"Rows remaining: {len(df_filtered)}")
print(f"Filtered dataset saved to:\n  {output_path}")