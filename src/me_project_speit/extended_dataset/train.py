"""Train the baseline MLP on an .npz dataset (X, T_final, split), using its saved split."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.me_project_speit.models.mlp import MLP
from src.me_project_speit.train import evaluate

PKG = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PKG / "datasets" / "data_test_extended_360.npz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def loader(X, Y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    X = d["X"].astype(np.float32)
    Y = d["T_final"].astype(np.float32)
    split = d["split"]
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}

    x_mean, x_std = X[idx["train"]].mean(0), X[idx["train"]].std(0)
    y_mean, y_std = Y[idx["train"]].mean(0), Y[idx["train"]].std(0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    Xs, Ys = (X - x_mean) / x_std, (Y - y_mean) / y_std

    train_loader = loader(Xs[idx["train"]], Ys[idx["train"]], args.batch_size, True)
    val_loader = loader(Xs[idx["val"]], Ys[idx["val"]], args.batch_size, False)
    test_loader = loader(Xs[idx["test"]], Ys[idx["test"]], args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    y_std_t = torch.as_tensor(y_std, device=device)

    run_dir = args.output_dir or (
        PKG / "runs" / f"mlp_trained_on_{args.data.stem}_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_path = run_dir / "best_model.pt"
    print(f"Device: {device} | train {len(idx['train'])}, val {len(idx['val'])}, "
          f"test {len(idx['test'])} | output {run_dir}")

    best_val, best_epoch, stale = float("inf"), 0, 0
    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse_scaled", "val_mse_scaled", "val_mae", "val_rmse"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * yb.numel()
                n += yb.numel()

            val_mse, val_mae, val_rmse = evaluate(model, val_loader, device, y_std_t)
            writer.writerow([epoch, total / n, val_mse, val_mae, val_rmse])

            if val_mse < best_val:
                best_val, best_epoch, stale = val_mse, epoch, 0
                torch.save({
                    "model_state_dict": {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()},
                    "x_mean": torch.from_numpy(x_mean.copy()),
                    "x_std": torch.from_numpy(x_std.copy()),
                    "y_mean": torch.from_numpy(y_mean.copy()),
                    "y_std": torch.from_numpy(y_std.copy()),
                    "dataset": str(args.data.resolve()),
                    "seed": args.seed,
                    "train_indices": torch.as_tensor(idx["train"]),
                    "val_indices": torch.as_tensor(idx["val"]),
                    "test_indices": torch.as_tensor(idx["test"]),
                    "epoch": epoch,
                }, ckpt_path)
            else:
                stale += 1

            if epoch == 1 or epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | val MAE {val_mae:.4f} °C")
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    _, test_mae, test_rmse = evaluate(model, test_loader, device, y_std_t)

    print(f"Best epoch: {best_epoch}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
