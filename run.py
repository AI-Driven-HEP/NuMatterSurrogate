from pathlib import Path
import argparse

from src.dataset import prepare_dataset
from src.train import train


ROOT = Path(__file__).resolve().parent


def main():

    parser = argparse.ArgumentParser(
        description="Train NuMatterSurrogate from a prepared sample."
    )

    parser.add_argument(
        "--sample",
        type=str,
        default="data/parameter_samples.npz",
        help="Path to the input .npz sample.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for processed datasets.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    sample_file = Path(args.sample)

    if not sample_file.is_absolute():
        sample_file = ROOT / sample_file

    data_dir = Path(args.data_dir)

    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir


    # --------------------------------------------------------
    # Step 2: Train
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Training model")
    print("=" * 60)

    train(
        train_file=data_dir / "train.pt",
        val_file=data_dir / "val.pt",
        grid_file=data_dir / "grid.pt",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()