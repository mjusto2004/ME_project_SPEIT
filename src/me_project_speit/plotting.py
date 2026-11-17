"""Plots of a finished training run (needs ``matplotlib``).

A run is a folder written by ``train``: ``best_model.pt`` and ``history.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def plot_run(
    run_dir: str | Path,
    sample: int = 0,
    half: bool = False,
    save: str | Path | None = None,
) -> list[Path]:
    """Plot a training run.

    Figure 1: training curves. Figure 2: for the test simulation number
    ``sample``, the true temperature, the prediction and the error at every
    node (in 3D when the node coordinates are known), plus true vs predicted,
    the error histogram and the error of every test simulation.

    Parameters
    ----------
    run_dir : folder of the run (``best_model.pt``, ``history.csv``).
    sample : which test simulation to draw (0 = first).
    half : draw only the nodes with y <= 0.5, to see inside the cube.
    save : if given, figures are saved as ``<save>_curves.png`` and
        ``<save>_mesh.png`` instead of being shown.

    Returns
    -------
    list of the saved files (empty when the figures are shown).
    """
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")  # no window needed
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    figures = {"curves": _curves(plt, run_dir, checkpoint)}

    figures["mesh"] = _mesh_prediction(plt, run_dir, checkpoint, sample, half)

    saved = []
    if save is None:
        plt.show()
    else:
        for name, fig in figures.items():
            path = Path(f"{save}_{name}.png")
            fig.savefig(path, dpi=130, bbox_inches="tight")
            saved.append(path)
    return saved


def _curves(plt, run_dir: Path, checkpoint: dict):
    """Training and validation error per epoch."""
    h = np.genfromtxt(run_dir / "history.csv", delimiter=",", names=True, ndmin=1)
    has_lr = "lr" in (h.dtype.names or ()) and np.isfinite(h["lr"]).any()
    fig, axes = plt.subplots(1, 3 if has_lr else 2, figsize=(15 if has_lr else 10, 4))
    a, b = axes[0], axes[1]

    a.plot(h["epoch"], h["train_mse_scaled"], label="train")
    a.plot(h["epoch"], h["val_mse_scaled"], label="validation")
    a.axvline(checkpoint["epoch"], color="gray", ls=":", label="best model")
    a.set(xlabel="epoch", ylabel="MSE (normalised data)", yscale="log", title="Loss")
    a.legend()

    b.plot(h["epoch"], h["val_mae"], color="C1")
    b.axvline(checkpoint["epoch"], color="gray", ls=":")
    b.set(xlabel="epoch", ylabel="validation MAE (°C)", title="Validation error")

    if has_lr:
        c = axes[2]
        c.plot(h["epoch"], h["lr"], color="C2")
        c.axvline(checkpoint["epoch"], color="gray", ls=":")
        c.set(xlabel="epoch", ylabel="learning rate", yscale="log", title="Learning rate (start of epoch)")

    fig.suptitle(f"{run_dir.name}  ({checkpoint['model_name']})")
    fig.tight_layout()
    return fig


def _mesh_prediction(plt, run_dir: Path, checkpoint: dict, sample: int, half: bool):
    """True / predicted / error on the nodes of one test simulation, parity plot,
    error histogram and error per test simulation (see :mod:`.evaluation`)."""
    from .evaluation import draw_evaluation, evaluate

    dataset_path = Path(checkpoint["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"The dataset used for this run is not at '{dataset_path}' any more."
        )
    ev = evaluate(run_dir / "best_model.pt", dataset_path, subset="test")
    if not 0 <= sample < len(ev.results):
        raise ValueError(f"--sample must be between 0 and {len(ev.results) - 1} (test simulations).")
    return draw_evaluation(plt, ev, ev.results[sample], half, label=f"test simulation {sample}")
