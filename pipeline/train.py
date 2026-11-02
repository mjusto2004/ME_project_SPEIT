
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP on EasyFEA simulations")

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, patience and lr must be positive")

    return args


def evaluate(model, loader, device, y_std):
    """Compute normalized MSE and real-unit MAE/RMSE."""
    model.eval()

    sq_norm = 0.0
    abs_real = 0.0
    sq_real = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            prediction = model(xb)
            difference = prediction - yb

            real_difference = difference * y_std

            sq_norm += difference.square().sum().item()
            abs_real += real_difference.abs().sum().item()
            sq_real += real_difference.square().sum().item()
            count += yb.numel()

    return (
        sq_norm / count,
        abs_real / count,
        math.sqrt(sq_real / count)
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------
    # Device
    # -------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Request a GPU through Slurm.")

    device = torch.device(
        "cuda" if (
            args.device == "cuda"
            or (args.device == "auto" and torch.cuda.is_available())
        ) else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------
    # Load dataset
    # -------------------------------
    data = np.loadtxt(args.dataset, delimiter=",", ndmin=2)

    if data.shape[1] != 563:
        raise ValueError(
            f"Expected 563 columns, got {data.shape[1]}"
        )

    if len(data) < 3:
        raise ValueError("At least 3 samples are required.")

    if not np.isfinite(data).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if len(data) < 50:
        print("Warning: dataset is too small for reliable evaluation.")

    X = data[:, :3].astype(np.float32)
    Y = data[:, 3:].astype(np.float32)

    # -------------------------------
    # Train / validation / test split
    # -------------------------------
    n = len(data)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    # -------------------------------
    # Normalization
    # -------------------------------
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)

    y_mean = Y[train_idx].mean(axis=0)
    y_std = Y[train_idx].std(axis=0)

    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)

    X_scaled = (X - x_mean) / x_std
    Y_scaled = (Y - y_mean) / y_std

    # -------------------------------
    # DataLoaders
    # -------------------------------
    def make_loader(indices, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(X_scaled[indices]),
            torch.from_numpy(Y_scaled[indices])
        )

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            pin_memory=device.type == "cuda"
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    # -------------------------------
    # Model and optimizer
    # -------------------------------
    model = MLP().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    y_std_tensor = torch.as_tensor(y_std, device=device)

    # -------------------------------
    # Output directory
    # -------------------------------
    run_dir = args.output_dir or (
        Path(__file__).resolve().parent
        / "runs"
        / f"{args.dataset.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "best_model.pt"

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {n}")
    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")

    # -------------------------------
    # Training
    # -------------------------------
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_mse_scaled",
            "val_mse_scaled",
            "val_mae",
            "val_rmse"
        ])

        for epoch in range(1, args.epochs + 1):

            model.train()

            train_sum = 0.0
            train_elements = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad()

                prediction = model(xb)
                loss = criterion(prediction, yb)

                loss.backward()
                optimizer.step()

                train_sum += loss.item() * yb.numel()
                train_elements += yb.numel()

            train_mse = train_sum / train_elements

            # Validation
            val_mse, val_mae, val_rmse = evaluate(
                model, val_loader, device, y_std_tensor
            )

            if not math.isfinite(val_mse):
                raise RuntimeError("Non-finite validation loss.")

            writer.writerow([
                epoch, train_mse, val_mse, val_mae, val_rmse
            ])
            f.flush()

            # Save best model
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save({
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "x_mean": torch.from_numpy(x_mean.copy()),
                    "x_std": torch.from_numpy(x_std.copy()),
                    "y_mean": torch.from_numpy(y_mean.copy()),
                    "y_std": torch.from_numpy(y_std.copy()),
                    "dataset": str(args.dataset.resolve()),
                    "seed": args.seed,
                    "train_indices": torch.as_tensor(train_idx.copy()),
                    "val_indices": torch.as_tensor(val_idx.copy()),
                    "test_indices": torch.as_tensor(test_idx.copy()),
                    "epoch": epoch,
                }, checkpoint_path)

            else:
                epochs_without_improvement += 1

            if (
                epoch == 1
                or epoch % 10 == 0
                or epochs_without_improvement >= args.patience
            ):
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train MSE: {train_mse:.5f} | "
                    f"Val MSE: {val_mse:.5f} | "
                    f"Val MAE: {val_mae:.3f}",
                    flush=True
                )

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # -------------------------------
    # Final test evaluation
    # -------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_mse, test_mae, test_rmse = evaluate(
        model, test_loader, device, y_std_tensor
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test normalized MSE: {test_mse:.5f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {run_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
