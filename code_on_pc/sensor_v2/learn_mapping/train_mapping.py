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
import sys
print(sys.executable)
import matplotlib
matplotlib.use("Qt5Agg")  # or "Agg" if you just want to save figures
print("Backend before importing pyplot:", matplotlib.get_backend())

import matplotlib.pyplot as plt
print("Backend after importing pyplot:", matplotlib.get_backend())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# --- Plot style settings ---
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

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

def tv_loss(t, mask=None):
    """
    t: (batch, N) flattened thickness (1024)
    mask: optional (N,) numpy or torch
    """
    B = t.size(0)
    t = t.view(B, 1, 32, 32)

    dx = t[:, :, :, 1:] - t[:, :, :, :-1]
    dy = t[:, :, 1:, :] - t[:, :, :-1, :]

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(
                mask, dtype=t.dtype, device=t.device
            )
        mask = mask.view(1, 1, 32, 32)
        dx = dx * mask[:, :, :, 1:]
        dy = dy * mask[:, :, 1:, :]

    return dx.abs().mean() + dy.abs().mean()


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

# ------------------- Training Loop -------------------
def train_model(train_df, val_df=None, epochs=30, batch_size=512, lr=1e-3, device='cpu', save_path="model.pth"):
    # Mask for thickness loss
    mask = make_circular_mask(32)
    print(mask.sum())
    #mask = torch.from_numpy(mask).to(device)

    # If no validation set is provided, split from train_df
    if val_df is None:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    train_ds = HallDataset(train_df)
    val_ds = HallDataset(val_df)
    # Before fitting scaler
    #train_ds.Y_thickness *= mask
    #val_ds.Y_thickness   *= mask
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Scale inputs and outputs
    in_scaler = StandardScaler().fit(train_ds.X)
    thickness_scaler = StandardScaler().fit(train_ds.Y_thickness)
    scalars_scaler = StandardScaler().fit(train_ds.Y_scalars)
    hall13_scaler = StandardScaler().fit(train_ds.Y_hall13)

    # Apply scaling
    train_ds.X = in_scaler.transform(train_ds.X).astype(np.float32)
    val_ds.X = in_scaler.transform(val_ds.X).astype(np.float32)
    train_ds.Y_thickness = thickness_scaler.transform(train_ds.Y_thickness).astype(np.float32)
    val_ds.Y_thickness = thickness_scaler.transform(val_ds.Y_thickness).astype(np.float32)
    train_ds.Y_scalars = scalars_scaler.transform(train_ds.Y_scalars).astype(np.float32)
    val_ds.Y_scalars = scalars_scaler.transform(val_ds.Y_scalars).astype(np.float32)
    train_ds.Y_hall13 = hall13_scaler.transform(train_ds.Y_hall13).astype(np.float32)
    val_ds.Y_hall13 = hall13_scaler.transform(val_ds.Y_hall13).astype(np.float32)

    # Model
    device = torch.device(device)
    model = MultiOutputMLP()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    train_losses, val_losses = [], []
    train_thickness, val_thickness = [], []
    train_scalars, val_scalars = [], []
    train_hall13, val_hall13 = [], []
        
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_loss_thickness = 0.0
        train_loss_scalars = 0.0
        train_loss_hall13 = 0.0

        for batch in train_loader:
            halls = batch['hall'].to(device)
            t_true = batch['thickness'].to(device)
            s_true = batch['scalars'].to(device)
            h13_true = batch['hall13'].to(device)

            optimizer.zero_grad()
            t_pred, s_pred, h13_pred = model(halls)

            # Compute losses
            loss_thickness = masked_mse(t_pred, t_true, mask)
            #loss_thickness = nn.MSELoss()(t_pred, t_true)
            #loss_tv = tv_loss(t_pred, mask)
            loss_scalars   = nn.MSELoss()(s_pred, s_true)
            loss_hall13    = nn.MSELoss()(h13_pred, h13_true)

            # Total loss
            loss = loss_thickness + 5*loss_scalars + loss_hall13
            loss.backward()
            optimizer.step()

            # Accumulate
            train_loss += loss.item() * halls.size(0)
            train_loss_thickness += loss_thickness.item() * halls.size(0)
            train_loss_scalars   += 5*loss_scalars.item()   * halls.size(0)
            train_loss_hall13    += loss_hall13.item()    * halls.size(0)

        # Normalize by dataset size
        train_loss /= len(train_ds)
        train_loss_thickness /= len(train_ds)
        train_loss_scalars   /= len(train_ds)
        train_loss_hall13    /= len(train_ds)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_loss_thickness = 0.0
        val_loss_scalars = 0.0
        val_loss_hall13 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                halls = batch['hall'].to(device)
                t_true = batch['thickness'].to(device)
                s_true = batch['scalars'].to(device)
                h13_true = batch['hall13'].to(device)

                t_pred, s_pred, h13_pred = model(halls)
                loss_thickness = masked_mse(t_pred, t_true, mask)
                #loss_thickness = nn.MSELoss()(t_pred, t_true)
                loss_scalars   = nn.MSELoss()(s_pred, s_true)
                loss_hall13    = nn.MSELoss()(h13_pred, h13_true)

                loss = loss_thickness + 5*loss_scalars + loss_hall13

                val_loss += loss.item() * halls.size(0)
                val_loss_thickness += loss_thickness.item() * halls.size(0)
                val_loss_scalars   += 5*loss_scalars.item()   * halls.size(0)
                val_loss_hall13    += loss_hall13.item()    * halls.size(0)

        val_loss /= len(val_ds)
        val_loss_thickness /= len(val_ds)
        val_loss_scalars   /= len(val_ds)
        val_loss_hall13    /= len(val_ds)


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_thickness.append(train_loss_thickness)
        val_thickness.append(val_loss_thickness)
        train_scalars.append(train_loss_scalars)
        val_scalars.append(val_loss_scalars)
        train_hall13.append(train_loss_hall13)
        val_hall13.append(val_loss_hall13)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Total: {train_loss:.6f} "
            f"(Thick: {train_loss_thickness:.6f}, Scalars: {train_loss_scalars:.6f}, Hall13: {train_loss_hall13:.6f}) | "
            f"Val Total: {val_loss:.6f} "
            f"(Thick: {val_loss_thickness:.6f}, Scalars: {val_loss_scalars:.6f}, Hall13: {val_loss_hall13:.6f})"
        )

        
            

    torch.save({
                'model_state_dict': model.state_dict(),
                'in_scaler': in_scaler,
                'thickness_scaler': thickness_scaler,
                'scalars_scaler': scalars_scaler,
                'hall13_scaler': hall13_scaler
            }, save_path)
    return model, in_scaler, thickness_scaler, scalars_scaler, hall13_scaler, {
            "train_total": train_losses,
            "val_total": val_losses,
            "train_thickness": train_thickness,
            "val_thickness": val_thickness,
            "train_scalars": train_scalars,
            "val_scalars": val_scalars,
            "train_hall13": train_hall13,
            "val_hall13": val_hall13
        }


# ------------------- Main -------------------
if __name__ == "__main__":

    save_path = "../model.pth" # change these paths to where the data is 
    # Load the CSVs
    csv_paths = ["../data_filtered_2.csv" ] # change these paths to where the data is 

    csv_val_paths = ["../data_filtered_2.csv"] # change these paths to where the data is 

    dfs = []
    for csv_path in csv_paths:
        assert os.path.exists(csv_path), f"{csv_path} not found!"
        dfs.append(pd.read_csv(csv_path))

    # Concatenate into one DataFrame
    df_all = pd.concat(dfs, ignore_index=True)
    print("num data points", len(df_all))

    dfs = []
    for csv_path in csv_val_paths:
        assert os.path.exists(csv_path), f"{csv_path} not found!"
        dfs.append(pd.read_csv(csv_path))

    # Concatenate into one DataFrame
    val_df = pd.concat(dfs, ignore_index=True)
    print("num data points validation", len(val_df))

    val_df = None
    

    # --- Extract history ---
    model, in_scaler, thickness_scaler, scalars_scaler, hall13_scaler, history = train_model(
        df_all, val_df=val_df, batch_size=512, lr=1e-4, epochs=25, device="cpu", save_path=save_path
    )

    # --- Plot total loss ---
    plt.figure(figsize=(6,4))
    plt.plot(history["train_total"], label="Train total loss")
    plt.plot(history["val_total"], label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.show()

    # --- Plot each component ---
    fig, axs = plt.subplots(3, 1, figsize=(6,8), sharex=True)
    axs[0].plot(history["train_thickness"], label="Train thickness")
    axs[0].plot(history["val_thickness"], label="Val thickness")
    axs[0].set_ylabel("Thickness loss")
    axs[0].legend()

    axs[1].plot(history["train_scalars"], label="Train scalars")
    axs[1].plot(history["val_scalars"], label="Val scalars")
    axs[1].set_ylabel("Scalars loss")
    axs[1].legend()

    axs[2].plot(history["train_hall13"], label="Train hall13")
    axs[2].plot(history["val_hall13"], label="Val hall13")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Hall13 loss")
    axs[2].legend()

    plt.tight_layout()
    plt.show()