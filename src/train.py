from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import ConditionalUResNet


# ============================================================
# Dataset
# ============================================================

class MatterDataset(Dataset):

    def __init__(self, filename):

        data = torch.load(
            filename,
            weights_only=True
        )

        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]



def train(train_file,
          val_file,
          grid_file,
          epochs,
          batch_size):
        
    # ============================================================
    # Configuration
    # ============================================================

    ROOT = Path(__file__).resolve().parents[1]

    CHECKPOINT_DIR = ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    # ============================================================
    # Data
    # ============================================================

    train_dataset = MatterDataset(train_file)
    val_dataset = MatterDataset(val_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Training samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Device             :", DEVICE)


    # ============================================================
    # Model
    # ============================================================

    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    # Initialize model
    model = ConditionalUResNet(
        in_channels=3,
        out_channels=3,
    ).to(DEVICE)



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    
    # Check if best checkpoint exists and load it
    if checkpoint_path.exists():
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        best_val_loss = checkpoint.get("val_loss", float('inf'))
        print(f"val_loss : {best_val_loss}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Model loaded successfully:")
        print(model)
    else:
        best_val_loss = float("inf")

        print(f"No existing checkpoint found at {checkpoint_path}")
        print("Starting training from scratch with new model:")
        print(model)



    # ============================================================
    # Loss / optimizer
    # ============================================================

    lambda_z = 0.01
    # ============================================================
    # Training
    # ============================================================

    for epoch in range(epochs):

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        model.train()

        train_loss = 0.0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y = y.permute(0, 3, 1, 2)
            
            optimizer.zero_grad()

            y_pred, z_pred, z_target = model(x,y)
            
            image_loss = F.mse_loss(y_pred, y)
            latent_loss = F.mse_loss(z_pred, z_target)

            loss = image_loss + lambda_z * latent_loss

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataset)

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y = y.permute(0, 3, 1, 2)
                
                y_pred, z_pred, z_target = model(x,y)

                image_loss = F.mse_loss(y_pred, y)
                latent_loss = F.mse_loss(z_pred, z_target)

                loss = image_loss + lambda_z * latent_loss

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train: {train_loss:.6e} | "
            f"Val: {val_loss:.6e} | "
            f"LR: {current_lr:.3e} | "
            f"image={image_loss.item():.5e} | "
            f"latent={latent_loss.item():.5e} | "
        )

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best_model.pt",
            )

            print(
                f"  -> saved best model "
                f"(val={val_loss:.6e})"
            )


    print()
    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6e}")