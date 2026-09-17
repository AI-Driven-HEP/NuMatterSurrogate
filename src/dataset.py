import numpy as np
import torch
from pathlib import Path


# ============================================================
# Normalization
# ============================================================

def normalize_parameters(theta12, dm21):
    """
    Normalize the parameters to approximately [-1, 1].

    theta12 : degrees
    dm21    : eV^2
    """

    theta12_min = 30.0
    theta12_max = 40.0

    dm21_min = 6.0e-5
    dm21_max = 9.0e-5

    theta12_norm = (
        2.0 * (theta12 - theta12_min)
        / (theta12_max - theta12_min)
        - 1.0
    )

    dm21_norm = (
        2.0 * (dm21 - dm21_min)
        / (dm21_max - dm21_min)
        - 1.0
    )

    return theta12_norm, dm21_norm


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(
    input_file,
    output_dir,
    val_fraction=0.2,
    seed=42,
):
    """
    Load the generated PEANUTS samples, normalize the physics
    parameters, split into train/validation sets, and save
    PyTorch-compatible files.
    """

    input_file = Path(input_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load generated data
    # --------------------------------------------------------

    data = np.load(input_file)

    theta12 = data["param"][:,0].astype(np.float32)
    dm21 = data["param"][:,1].astype(np.float32)

    U_Evol = data["U_Evol"].astype(np.float32)

    print("Loaded:")
    print("theta12 :", theta12.shape)
    print("dm21    :", dm21.shape)
    print("U_Evol  :", U_Evol.shape)

    # --------------------------------------------------------
    # Normalize physics parameters
    # --------------------------------------------------------

    theta12_norm, dm21_norm = normalize_parameters(
        theta12,
        dm21,
    )

    X = np.stack(
        [theta12_norm, dm21_norm],
        axis=1,
    )

    # --------------------------------------------------------
    # Random train / validation split
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    n_samples = len(theta12)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(val_fraction * n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "val_indices.npy", val_indices)

    # --------------------------------------------------------
    # Delta normalization: standardization + sigmoid
    # --------------------------------------------------------

    train_delta = U_Evol[train_indices]

    delta_mean = train_delta.mean(axis=0)   # (H,W,C)
    delta_std  = train_delta.std(axis=0)    # (H,W,C)

    delta_std = np.maximum(delta_std, 1e-12)

    # Save normalization parameters
    np.savez(
        output_dir / "delta_normalization.npz",
        mean=delta_mean,
        std=delta_std,
    )

    Delta_standardized = (
        U_Evol - delta_mean[None, :, :, :]
    ) / delta_std[None, :, :, :]

    # --------------------------------------------------------
    # Create PyTorch tensors
    # --------------------------------------------------------

    Delta_norm = torch.from_numpy(Delta_standardized.astype(np.float32))
    X = torch.from_numpy(X)
   
    train_data = {
        "x": X[train_indices],
        "y": Delta_norm[train_indices],
    }

    val_data = {
        "x": X[val_indices],
        "y": Delta_norm[val_indices],
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        train_data,
        output_dir / "train.pt",
    )

    torch.save(
        val_data,
        output_dir / "val.pt",
    )

    print()
    print(f"Training samples   : {len(train_indices)}")
    print(f"Validation samples : {len(val_indices)}")
    print()
    print(f"Saved to: {output_dir}")


    # --------------------------------------------------------
    # Save normalization parameters
    # --------------------------------------------------------

    normalization_file = output_dir / "normalization.txt"

    with open(normalization_file, "w") as f:

        f.write("# Parameter normalization\n")
        f.write("theta12_min = 30.0\n")
        f.write("theta12_max = 40.0\n")
        f.write("dm21_min = 6.0e-5\n")
        f.write("dm21_max = 9.0e-5\n")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]

    prepare_dataset(
        input_file=ROOT / "data" / "parameter_samples.npz",
        output_dir=ROOT / "data",
        val_fraction=0.2,
        seed=42,
    )