"""Training run orchestration: data, split, normalisation, checkpoint, final test.

The training loop itself belongs to the model (:meth:`SurrogateModel.fit`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .datasets import DEFAULT_DATA_DIR, load_dataset, load_spec, resolve_dataset
from .models import build_model, check_compatible
from .utils import select_device, timestamp, to_plain


@dataclass
class TrainConfig:
    """Hyper-parameters and options of a training run."""

    dataset: Path  # path, or bare name of a CSV in ``data_dir``
    model: str = "mlp"
    model_kwargs: dict = field(default_factory=dict)  # constructor arguments, e.g. {"hidden": [64, 64]}
    fit_kwargs: dict = field(default_factory=dict)  # model-specific options of fit(), e.g. {"weight_decay": 1e-4}
    data_dir: Path = DEFAULT_DATA_DIR
    epochs: int = 300
    batch_size: int | None = None  # None: the dataset kind's default
    lr: float = 1e-3
    patience: int = 30
    seed: int = 42
    device: str = "auto"  # "auto", "cuda" or "cpu"
    output_dir: Path | None = None
    test_dataset: Path | None = None  # extra test set (same mesh and inputs), e.g. independent runs

    def __post_init__(self):
        self.dataset = Path(self.dataset)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.test_dataset is not None:
            self.test_dataset = Path(self.test_dataset)
        if (
            self.epochs < 1
            or (self.batch_size is not None and self.batch_size < 1)
            or self.lr <= 0
            or self.patience < 1
        ):
            raise ValueError("epochs, batch_size, patience and lr must be positive")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be 'auto', 'cuda' or 'cpu'")


def _check_same_data(spec, other, name: str) -> None:
    """The extra test set must have the same inputs and outputs as the training set."""
    if (other.kind, other.input_names, other.n_outputs, other.mesh_id) != (
        spec.kind, spec.input_names, spec.n_outputs, spec.mesh_id,
    ):
        raise ValueError(
            f"Test dataset '{name}' does not match the training data: "
            f"{other.describe()} vs {spec.describe()}"
        )


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
        ``checkpoint`` and ``history`` paths; plus ``independent_mse``,
        ``independent_mae`` and ``independent_rmse`` when ``cfg.test_dataset`` is given.
    """
    # Fail fast, before any heavy work, if the pair is incompatible.
    cfg.dataset = resolve_dataset(cfg.dataset, cfg.data_dir)
    check_compatible(cfg.model, load_spec(cfg.dataset))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = select_device(cfg.device)

    # --- Data ---------------------------------------------------------------
    data = load_dataset(cfg.dataset)
    spec = data.spec
    X, Y = data.X, data.Y
    n = len(data)  # number of simulations

    batch_size = cfg.batch_size or data.default_batch_size

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

    # Optional independent test set, normalised with the training statistics.
    extra_loader = extra = None
    if cfg.test_dataset is not None:
        extra_path = resolve_dataset(cfg.test_dataset, cfg.data_dir)
        extra = load_dataset(extra_path)
        _check_same_data(spec, extra.spec, extra_path.name)
        extra_loader = extra.make_loader(
            (extra.X - x_mean) / x_std, (extra.Y - y_mean) / y_std,
            np.arange(len(extra.X)), batch_size, False, pin_memory=device.type == "cuda",
        )

    # --- Model --------------------------------------------------------------
    model = build_model(cfg.model, spec, **cfg.model_kwargs).to(device)
    y_std_tensor = torch.as_tensor(y_std, device=device)

    run_dir = cfg.output_dir or (
        Path("runs") / f"{cfg.dataset.stem}_{timestamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"

    # --- Summary: general, then model, then training ------------------------
    device_name = f"{device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    print(f"Model:  {cfg.model}")
    print(f"Device:  {device_name}")
    print()
    print("Data")
    print(f"  Dataset:     {cfg.dataset} ({spec.describe()})")
    print(f"  Simulations: {n} ({len(X)} rows)")
    print(f"  Split:       {data.count_simulations(train_idx)} train / "
          f"{data.count_simulations(val_idx)} validation / {data.count_simulations(test_idx)} test")
    if extra is not None:
        print(f"  Extra test:  {extra_path} ({len(extra)} simulations)")
    if n < 50:
        print("  Warning: dataset is too small for reliable evaluation.")
    print()
    print("Model")
    print(f"  Options:     {cfg.model_kwargs or 'defaults'}")
    print(f"  Parameters:  {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("Training")
    print(f"  Epochs:      {cfg.epochs} (patience {cfg.patience})")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {cfg.lr}")
    print(f"  Seed:        {cfg.seed}")
    print(f"  Options:     {cfg.fit_kwargs or 'defaults'}")
    print()

    # --- Training (the model owns its loop) ---------------------------------
    def save_checkpoint(epoch: int) -> None:
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "model_name": cfg.model,
                "model_kwargs": to_plain(cfg.model_kwargs),
                "fit_kwargs": to_plain(cfg.fit_kwargs),
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

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse_scaled", "val_mse_scaled", "val_mae", "val_rmse"])

        def log_epoch(*row) -> None:
            writer.writerow(row)
            f.flush()

        best_epoch = model.fit(
            train_loader, val_loader,
            epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience,
            device=device, y_std=y_std_tensor,
            on_epoch=log_epoch, on_best=save_checkpoint, **cfg.fit_kwargs,
        )

    # --- Final test evaluation with the best checkpoint ---------------------
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_mse, test_mae, test_rmse = model.evaluate(test_loader, device, y_std_tensor)

    print(f"Best epoch: {best_epoch}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test normalized MSE: {test_mse:.5f}")
    result = {}
    if extra_loader is not None:
        ind_mse, ind_mae, ind_rmse = model.evaluate(extra_loader, device, y_std_tensor)
        print(f"Independent test MAE: {ind_mae:.4f}")
        print(f"Independent test RMSE: {ind_rmse:.4f}")
        result = {"independent_mse": ind_mse, "independent_mae": ind_mae, "independent_rmse": ind_rmse}
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")

    return {
        "best_epoch": best_epoch,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "checkpoint": checkpoint_path,
        "history": history_path,
        **result,
    }
