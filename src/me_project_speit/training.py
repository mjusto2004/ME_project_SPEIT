"""Training and evaluation of the MLP surrogate model."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .datasets import DEFAULT_DATA_DIR, load_dataset, load_spec, resolve_dataset
from .models import build_model, check_compatible


@dataclass
class TrainConfig:
    """Hyper-parameters and options of a training run."""

    dataset: Path  # path, or bare name of a CSV in ``data_dir``
    model: str = "mlp"
    model_kwargs: dict = field(default_factory=dict)  # e.g. {"hidden": [64, 64]}
    data_dir: Path = DEFAULT_DATA_DIR
    epochs: int = 300
    batch_size: int | None = None  # None: the dataset kind's default
    lr: float = 1e-3
    patience: int = 30
    seed: int = 42
    device: str = "auto"  # "auto", "cuda" or "cpu"
    output_dir: Path | None = None

    def __post_init__(self):
        self.dataset = Path(self.dataset)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if (
            self.epochs < 1
            or (self.batch_size is not None and self.batch_size < 1)
            or self.lr <= 0
            or self.patience < 1
        ):
            raise ValueError("epochs, batch_size, patience and lr must be positive")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be 'auto', 'cuda' or 'cpu'")


def select_device(name: str) -> torch.device:
    """Resolve ``"auto" | "cuda" | "cpu"`` to a :class:`torch.device`."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Request a GPU through Slurm.")
    use_cuda = name == "cuda" or (name == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def evaluate(model, loader, device, y_std):
    """Compute the normalized MSE and the real-unit MAE / RMSE.

    ``y_std`` converts normalized differences back to degrees.
    """
    model.eval()

    sq_norm = abs_real = sq_real = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            *inputs, yb = batch  # the last element is the target
            inputs = [t.to(device, non_blocking=True) for t in inputs]
            yb = yb.to(device, non_blocking=True)

            difference = model(*inputs) - yb
            real_difference = difference * y_std

            sq_norm += difference.square().sum().item()
            abs_real += real_difference.abs().sum().item()
            sq_real += real_difference.square().sum().item()
            count += yb.numel()

    return sq_norm / count, abs_real / count, math.sqrt(sq_real / count)


def train(cfg: TrainConfig) -> dict:
    """Train the model ``cfg.model`` on ``cfg.dataset`` and save the best checkpoint.

    Model/dataset compatibility is checked first, so an invalid pair fails
    immediately with an explicit message. The run directory receives
    ``best_model.pt`` (weights, model name and arguments, dataset spec,
    normalisation statistics, split indices) and ``history.csv``.

    Returns
    -------
    dict
        ``best_epoch``, ``test_mse``, ``test_mae``, ``test_rmse``,
        ``checkpoint`` and ``history`` paths.
    """
    # Fail fast, before any heavy work, if the pair is incompatible.
    cfg.dataset = resolve_dataset(cfg.dataset, cfg.data_dir)
    check_compatible(cfg.model, load_spec(cfg.dataset))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = select_device(cfg.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---------------------------------------------------------------
    data = load_dataset(cfg.dataset)
    spec = data.spec
    X, Y = data.X, data.Y
    n = len(data)  # number of simulations
    if n < 50:
        print("Warning: dataset is too small for reliable evaluation.")

    batch_size = cfg.batch_size or data.default_batch_size
    print(f"Batch size: {batch_size}")

    split = data.split(cfg.seed)
    train_idx, val_idx, test_idx = split.train, split.val, split.test

    # Normalisation statistics come from the training set only.
    x_mean, x_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    y_mean, y_std = Y[train_idx].mean(axis=0), Y[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)

    X_scaled = (X - x_mean) / x_std
    Y_scaled = (Y - y_mean) / y_std

    def make_loader(indices, shuffle):
        # Each kind of dataset knows how to build its batches (rows, whole graphs...).
        return data.make_loader(
            X_scaled, Y_scaled, indices, batch_size, shuffle,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    # --- Model --------------------------------------------------------------
    model = build_model(cfg.model, spec, **cfg.model_kwargs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    y_std_tensor = torch.as_tensor(y_std, device=device)

    run_dir = cfg.output_dir or (
        Path("runs") / f"{cfg.dataset.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"

    print(f"Model: {cfg.model} {cfg.model_kwargs or ''}")
    print(f"Dataset: {cfg.dataset} ({spec.describe()})")
    print(f"Simulations: {n} ({len(X)} rows)")
    print(f"Train: {data.count_simulations(train_idx)}")
    print(f"Validation: {data.count_simulations(val_idx)}")
    print(f"Test: {data.count_simulations(test_idx)}")

    # --- Training loop ------------------------------------------------------
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_mse_scaled", "val_mse_scaled", "val_mae", "val_rmse"]
        )

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_sum = 0.0
            train_elements = 0

            for batch in train_loader:
                *inputs, yb = batch  # the last element is the target
                inputs = [t.to(device, non_blocking=True) for t in inputs]
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad()
                loss = criterion(model(*inputs), yb)
                loss.backward()
                optimizer.step()

                train_sum += loss.item() * yb.numel()
                train_elements += yb.numel()

            train_mse = train_sum / train_elements
            val_mse, val_mae, val_rmse = evaluate(model, val_loader, device, y_std_tensor)

            if not math.isfinite(val_mse):
                raise RuntimeError("Non-finite validation loss.")

            writer.writerow([epoch, train_mse, val_mse, val_mae, val_rmse])
            f.flush()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                        "model_name": cfg.model,
                        "model_kwargs": _plain(cfg.model_kwargs),
                        "spec": spec.to_dict(),
                        "x_mean": torch.from_numpy(x_mean.copy()),
                        "x_std": torch.from_numpy(x_std.copy()),
                        "y_mean": torch.from_numpy(y_mean.copy()),
                        "y_std": torch.from_numpy(y_std.copy()),
                        "dataset": str(cfg.dataset.resolve()),
                        "seed": cfg.seed,
                        "train_indices": torch.as_tensor(train_idx.copy()),
                        "val_indices": torch.as_tensor(val_idx.copy()),
                        "test_indices": torch.as_tensor(test_idx.copy()),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if (
                epoch == 1
                or epoch % 10 == 0
                or epochs_without_improvement >= cfg.patience
            ):
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train MSE: {train_mse:.5f} | "
                    f"Val MSE: {val_mse:.5f} | "
                    f"Val MAE: {val_mae:.3f}",
                    flush=True,
                )

            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- Final test evaluation with the best checkpoint ---------------------
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, y_std_tensor)

    print(f"Best epoch: {best_epoch}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test normalized MSE: {test_mse:.5f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")

    return {
        "best_epoch": best_epoch,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "checkpoint": checkpoint_path,
        "history": history_path,
    }


def _plain(obj):
    """Convert tuples to lists so checkpoints load with ``weights_only=True``."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj
