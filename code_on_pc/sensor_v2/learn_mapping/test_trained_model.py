import pandas as pd
import numpy as np
import ast
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ------------------- Helpers -------------------
def parse_hall(hall_str):
    """Parse hall_data string into numpy array"""
    return np.array(ast.literal_eval(hall_str), dtype=float)


def downsample_thickness_map(thickness_str, target_size=(32,32)):
    """
    Parse thickness_map string, resize to target_size using bilinear interpolation.
    Uses the full original map, no cropping.
    """
    from scipy.ndimage import zoom

    tmap = np.array(ast.literal_eval(thickness_str), dtype=float)
    h, w = tmap.shape
    th, tw = target_size

    # Compute zoom factors to exactly match target size
    zoom_h = th / h
    zoom_w = tw / w

    # Bilinear interpolation (order=1)
    t_resized = zoom(tmap, (zoom_h, zoom_w), order=1)

    # Normalize so sum = 1
    total = t_resized.sum()
    if total > 0:
        t_resized /= total

    return t_resized.flatten()


def make_circular_mask(size=32):
    """Return a (size*size,) mask with 1 inside circle, 0 outside."""
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = size / 2
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = (dist <= radius).astype(np.float32)
    return mask.flatten()


def masked_mse(pred, target, mask):
    # pred, target: (batch, N)
    # mask: (N,)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=pred.dtype, device=pred.device)
    diff = (pred - target) * mask
    return (diff**2).sum() / (mask.sum() * pred.size(0))


# ------------------- PyTorch Dataset -------------------
class HallDataset(Dataset):
    def __init__(self, df, input_sensors=12, thickness_size=(32,32)):
        self.X = []
        self.Y_thickness = []
        self.Y_scalars = []
        self.Y_hall13 = []
        for _, row in df.iterrows():
            hall = parse_hall(row['hall_data'])
            # input: first 12 sensors
            self.X.append(hall[:input_sensors*3])
            # target hall13
            self.Y_hall13.append(hall[input_sensors*3:input_sensors*3+3])
            # thickness map downsampled
            self.Y_thickness.append(
                downsample_thickness_map(row['thickness_map'], target_size=thickness_size)
            )
            # forces/torques
            scalars = row[['fx','fy','fz','tx','ty','tz']].values.astype(float)
            self.Y_scalars.append(scalars)

        self.X = np.vstack(self.X).astype(np.float32)
        self.Y_thickness = np.vstack(self.Y_thickness).astype(np.float32)
        self.Y_scalars = np.vstack(self.Y_scalars).astype(np.float32)
        self.Y_hall13 = np.vstack(self.Y_hall13).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'hall': torch.from_numpy(self.X[idx]),
            'thickness': torch.from_numpy(self.Y_thickness[idx]),
            'scalars': torch.from_numpy(self.Y_scalars[idx]),
            'hall13': torch.from_numpy(self.Y_hall13[idx])
        }

# ------------------- Neural Network -------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, in_dim=36, hidden_dim=256, scalar_dim=6, hall13_dim=3):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Thickness head (outputs 32x32 map)
        self.thickness_head = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),          # (batch, 64, 4, 4)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # → 32x32
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),                       # (batch, 1, 32, 32)
        )

        # Scalars + hall13
        self.scalars_head = nn.Linear(hidden_dim, scalar_dim)
        self.hall13_head = nn.Linear(hidden_dim, hall13_dim)
        
    def forward(self, x):
        z = self.encoder(x)

        # Thickness prediction
        t_pred = self.thickness_head(z)              # (batch, 1, 32, 32)
        t_pred = t_pred.view(z.size(0), -1)          # flatten → (batch, 1024)

        # Scalars and hall13
        s_pred = self.scalars_head(z)
        h13_pred = self.hall13_head(z)

        return t_pred, s_pred, h13_pred

# ------------------- Test -------------------
def test_model(model_path, test_df, batch_size=32, device='cpu', fz_threshold=None):
    """
    Evaluate the model on test_df and print per-component statistics:
      - mean error (mu)
      - standard deviation of error (sigma)
      - normalized RMSE (% of tested range)
      - tested range
    Optionally filter rows where |fz| < fz_threshold.
    """
    # -------- Optional filtering by |fz| --------
    if fz_threshold is not None:
        before = len(test_df)
        test_df = test_df[np.abs(test_df["fz"]) >= fz_threshold].reset_index(drop=True)
        print(f"Filtered test data: {before} → {len(test_df)} (|fz| >= {fz_threshold})")

    # Mask for thickness loss
    mask = make_circular_mask(32)

    test_ds = HallDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device(device)
    model = MultiOutputMLP()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scalers
    in_scaler = checkpoint["in_scaler"]
    thickness_scaler = checkpoint["thickness_scaler"]
    scalars_scaler = checkpoint["scalars_scaler"]
    hall13_scaler = checkpoint["hall13_scaler"]

    # Apply scaling
    test_ds.X = in_scaler.transform(test_ds.X).astype(np.float32)
    test_ds.Y_thickness = thickness_scaler.transform(test_ds.Y_thickness).astype(np.float32)
    test_ds.Y_scalars = scalars_scaler.transform(test_ds.Y_scalars).astype(np.float32)
    test_ds.Y_hall13 = hall13_scaler.transform(test_ds.Y_hall13).astype(np.float32)

    # -------- Validation --------
    val_loss = 0.0
    val_loss_thickness = 0.0
    val_loss_scalars = 0.0
    val_loss_hall13 = 0.0

    all_pred_scalars = []
    all_true_scalars = []

    with torch.no_grad():
        for batch in test_loader:
            halls = batch["hall"].to(device)
            t_true = batch["thickness"].to(device)
            s_true = batch["scalars"].to(device)
            h13_true = batch["hall13"].to(device)

            t_pred, s_pred, h13_pred = model(halls)

            loss_thickness = masked_mse(t_pred, t_true, mask)
            loss_scalars = nn.MSELoss()(s_pred, s_true)
            loss_hall13 = nn.MSELoss()(h13_pred, h13_true)

            loss = loss_thickness + 5 * loss_scalars + loss_hall13

            val_loss += loss.item() * halls.size(0)
            val_loss_thickness += loss_thickness.item() * halls.size(0)
            val_loss_scalars += 5 * loss_scalars.item() * halls.size(0)
            val_loss_hall13 += loss_hall13.item() * halls.size(0)

            all_pred_scalars.append(s_pred.cpu().numpy())
            all_true_scalars.append(s_true.cpu().numpy())

    val_loss /= len(test_ds)
    val_loss_thickness /= len(test_ds)
    val_loss_scalars /= len(test_ds)
    val_loss_hall13 /= len(test_ds)

    print(
        f"Test Total: {val_loss:.6f} "
        f"(Thick: {val_loss_thickness:.6f}, "
        f"Scalars: {val_loss_scalars:.6f}, "
        f"Hall13: {val_loss_hall13:.6f})"
    )

    # -------- Force/Torque statistics --------
    all_pred_scalars = np.vstack(all_pred_scalars)
    all_true_scalars = np.vstack(all_true_scalars)

    # Convert back to physical units
    all_pred_scalars = scalars_scaler.inverse_transform(all_pred_scalars)
    all_true_scalars = scalars_scaler.inverse_transform(all_true_scalars)

    # Error in physical units
    errors = all_pred_scalars - all_true_scalars

    # Statistics
    mu = np.mean(errors, axis=0)
    sigma = np.std(errors, axis=0)
    mse = np.mean(errors**2, axis=0)
    rmse = np.sqrt(mse)

    min_vals = np.min(all_true_scalars, axis=0)
    max_vals = np.max(all_true_scalars, axis=0)
    ranges = max_vals - min_vals

    # Normalized RMSE (% of tested range)
    nrmse = 100 * rmse / ranges

    components = [r"$f_x$", r"$f_y$", r"$f_z$", r"$t_x$", r"$t_y$", r"$t_z$"]

    print("\nPer-component statistics:")
    print(
        f"{'Type':<6} {'mu':>12} {'sigma':>12} {'NRMSE (%)':>12} {'Range':>24}"
    )

    for c, m, s, nr, rmin, rmax in zip(
        components, mu, sigma, nrmse, min_vals, max_vals
    ):
        print(
            f"{c:<6} "
            f"{m:12.4e} "
            f"{s:12.4e} "
            f"{nr:12.2f} "
            f"[{rmin:7.3f}, {rmax:7.3f}]"
        )

    # -------- LaTeX table --------
    print("\n--- LaTeX Table ---\n")
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\begin{tabular}{c|c|c|c|c}")
    print("        \\hline")
    print("        Type & $\\mu$ & $\\sigma$ & NRMSE (\\%) & Range \\\\ \\hline")

    for c, m, s, nr, rmin, rmax in zip(
        components, mu, sigma, nrmse, min_vals, max_vals
    ):
        print(
            f"        {c} & "
            f"{m:.3e} & "
            f"{s:.3e} & "
            f"{nr:.2f} & "
            f"[{rmin:.3f}, {rmax:.3f}] \\\\ \\hline"
        )

    print("    \\end{tabular}")
    print("    \\caption{Force and torque estimation error statistics. "
          "NRMSE is normalized by the tested range.}")
    print("    \\label{tab:force_torque}")
    print("\\end{table}")

    return {
        "mu": mu,
        "sigma": sigma,
        "rmse": rmse,
        "nrmse": nrmse,
        "ranges": np.column_stack((min_vals, max_vals)),
    }


# ------------------- Main -------------------
if __name__ == "__main__":
    # Load the CSVs
    
    model_path = "../best_model.pth"
    csv_test_paths = ["../data_filtered.csv"
    ]

    dfs = []
    for csv_path in csv_test_paths:
        assert os.path.exists(csv_path), f"{csv_path} not found!"
        dfs.append(pd.read_csv(csv_path))

    # Concatenate into one DataFrame
    test_df = pd.concat(dfs, ignore_index=True)
    print("num data points test", len(test_df))
    test_model(model_path, test_df, batch_size=32, device="cpu", fz_threshold=0.1)
    test_model(model_path, test_df, batch_size=32, device="cpu", fz_threshold=None)
