import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
# Set seeds for reproducibility
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import matplotlib as mpl
import seaborn as sns
mpl.use("Agg")   # headless-safe backend

# --- Plot style settings ---
mpl.rcParams['pdf.fonttype'] = 42     # Embed TrueType fonts
mpl.rcParams['ps.fonttype']  = 42

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['Times New Roman']

mpl.rcParams['mathtext.fontset'] = 'dejavuserif'  # NOT 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})


# Reuse your existing imports / helper functions
from test_trained_model import (
    HallDataset,
    MultiOutputMLP,
    make_circular_mask,
    masked_mse,
    parse_hall,
    downsample_thickness_map
)

def visualize_pressure_maps(model_path, test_df, num_samples=4, device='cpu', fz_threshold=None):
    """
    Plots random predicted vs true thickness maps with masked MSE annotations.
    Produces a 4x2 grid (top: predicted, bottom: true).
    Only considers samples where |fz| >= fz_threshold if specified.
    """

    # -------- Optional filtering by |fz| --------
    if fz_threshold is not None:
        before = len(test_df)
        test_df = test_df[np.abs(test_df["fz"]) >= fz_threshold].reset_index(drop=True)
        print(f"Filtered test data: {before} → {len(test_df)} (|fz| >= {fz_threshold})")

    # -------- Dataset setup --------
    mask = make_circular_mask(32)
    test_ds = HallDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

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

    # Apply input scaling
    test_ds.X = in_scaler.transform(test_ds.X).astype(np.float32)
    test_ds.Y_thickness = thickness_scaler.transform(test_ds.Y_thickness).astype(np.float32)

    # -------- Pick random samples --------
    num_available = len(test_ds)
    if num_available == 0:
        print("No samples available after filtering.")
        return
    if num_samples > num_available:
        num_samples = num_available

    indices = random.sample(range(num_available), num_samples)
    print(f"Selected sample indices: {indices}")

    # -------- Plot --------
    fig, axes = plt.subplots(2, num_samples, figsize=(1.8*num_samples, 4))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i, idx in enumerate(indices):
        sample = test_ds[idx]
        hall = sample['hall'].unsqueeze(0).to(device)
        true_map = sample['thickness'].view(32, 32).cpu().numpy()

        with torch.no_grad():
            pred_map, _, _ = model(hall)
            #pred_map = pred_map.view(32, 32).cpu().numpy()

        # Inverse transform back to physical scale
        pred_unscaled = thickness_scaler.inverse_transform(pred_map.numpy())
        true_unscaled = thickness_scaler.inverse_transform(true_map.flatten()[None, :])[0].reshape(32, 32)

        # Compute masked MSE (flatten before passing to masked_mse)
        pred_flat = torch.tensor(pred_map.flatten()[None, :])
        true_flat = torch.tensor(true_map.flatten()[None, :])

        mse_val = masked_mse(pred_flat, true_flat, mask).item()

        pred_unscaled = pred_unscaled*mask

        pred_unscaled = pred_unscaled.reshape(32, 32).astype(np.float32)
        pred_unscaled = np.maximum(pred_unscaled, 0)
        pred_sum = np.sum(pred_unscaled)
        if pred_sum > 0:
            pred_unscaled = pred_unscaled/pred_sum

        # --- Shared color scale between true and predicted ---
        vmin = min(np.min(true_unscaled), np.min(pred_unscaled))
        vmax = max(np.max(true_unscaled), np.max(pred_unscaled))
        # Plot predicted
        im_pred = axes[0, i].imshow(pred_unscaled, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Pred #{idx}\nMSE={mse_val:.3e}")
        axes[0, i].axis('off')

        print(np.sum(pred_unscaled), np.sum(true_unscaled))
        
        
        # Plot true
        im_true = axes[1, i].imshow(true_unscaled, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')


    # --- Add single colorbar to the rightmost column ---
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im_pred, cax=cbar_ax)
    axes[0, 0].axis('on')
    axes[1, 0].axis('on')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[0, 0].set_ylabel("Predicted", fontsize=12)
    axes[1, 0].set_ylabel("True", fontsize=12)

    plt.suptitle("Predicted vs. True Normalized Mesh Overlap", fontsize=14)
    #plt.tight_layout()
    plt.savefig("pressure_map.pdf")
    #plt.show()


if __name__ == "__main__":
    # Example usage
    model_path = "../best_model.pth" # change these paths to where the data is 
    csv_test_path = "../data_filtered_2.csv" # change these paths to where the data is 

    assert os.path.exists(csv_test_path), f"{csv_test_path} not found!"
    test_df = pd.read_csv(csv_test_path)
    print("num data points test", len(test_df))

    visualize_pressure_maps(
        model_path,
        test_df,
        num_samples=4,
        device="cpu",
        fz_threshold=3.0,   # ← Set your min |fz| here
    )
