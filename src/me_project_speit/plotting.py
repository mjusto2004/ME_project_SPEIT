"""Plots of a finished training run (needs ``matplotlib``).

A run is a folder written by ``train``: ``best_model.pt`` and ``history.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def latest_run(runs_dir: str | Path = "runs") -> Path:
    """The most recently modified run folder in ``runs_dir``."""
    runs = [p for p in Path(runs_dir).glob("*") if (p / "best_model.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No run (folder with best_model.pt) found in '{runs_dir}'.")
    return max(runs, key=lambda p: (p / "best_model.pt").stat().st_mtime)


def plot_run(
    run_dir: str | Path,
    sample: int = 0,
    half: bool = False,
    save: str | Path | None = None,
) -> list[Path]:
    """Plot a training run.

    Figure 1 (any model): training curves. Figure 2 (``graph`` and ``pointwise`` datasets): for
    the test simulation number ``sample``, the true temperature, the
    prediction and the error at every node, plus true vs predicted for all
    test simulations.

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

    if checkpoint["spec"]["kind"] in ("graph", "pointwise"):
        figures["mesh"] = _mesh_prediction(plt, run_dir, checkpoint, sample, half)
    else:
        print("Only the training curves are drawn: 3D plots exist for 'graph' and 'pointwise' datasets.")

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
    h = np.genfromtxt(run_dir / "history.csv", delimiter=",", names=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))

    a.plot(h["epoch"], h["train_mse_scaled"], label="train")
    a.plot(h["epoch"], h["val_mse_scaled"], label="validation")
    a.axvline(checkpoint["epoch"], color="gray", ls=":", label="best model")
    a.set(xlabel="epoch", ylabel="MSE (normalised data)", yscale="log", title="Loss")
    a.legend()

    b.plot(h["epoch"], h["val_mae"], color="C1")
    b.axvline(checkpoint["epoch"], color="gray", ls=":")
    b.set(xlabel="epoch", ylabel="validation MAE (°C)", title="Validation error")

    fig.suptitle(f"{run_dir.name}  ({checkpoint['model_name']})")
    fig.tight_layout()
    return fig


def _mesh_prediction(plt, run_dir: Path, checkpoint: dict, sample: int, half: bool):
    """True / predicted / error on the nodes of one test mesh, and parity plot.

    For ``graph`` and ``pointwise`` datasets, whose node coordinates are inputs.
    """
    from .datasets import load_dataset
    from .inference import Predictor

    dataset_path = Path(checkpoint["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"The dataset used for this run is not at '{dataset_path}' any more."
        )
    data = load_dataset(dataset_path)
    predictor = Predictor(run_dir / "best_model.pt")

    names = data.spec.input_names
    test_rows = checkpoint["test_indices"].numpy()

    # Per kind of dataset: which simulations are in the test set, and how to get
    # the node inputs, true values and predictions of one of them.
    if checkpoint["spec"]["kind"] == "graph":
        test_sims = sorted(set(data.sim[test_rows].tolist()))

        def predict(s):
            a, b = data.node_start[s], data.node_stop[s]
            c, d = data.edge_start[s], data.edge_stop[s]
            edges = np.stack([data.edge_src[c:d], data.edge_dst[c:d]])
            return data.X[a:b], data.Y[a:b, 0], predictor.predict(data.X[a:b], edges)[:, 0]

    else:  # pointwise: the rows of simulation s are contiguous, rows_per_sample each
        n = data.spec.rows_per_sample
        test_sims = sorted(set((test_rows // n).tolist()))

        def predict(s):
            rows = slice(s * n, (s + 1) * n)
            return data.X[rows], data.Y[rows, 0], predictor.predict(data.X[rows])[:, 0]

    if not 0 <= sample < len(test_sims):
        raise ValueError(f"--sample must be between 0 and {len(test_sims) - 1} (test simulations).")

    fig = plt.figure(figsize=(15, 8))

    # --- one test mesh: true, predicted, error -------------------------------
    x, true, pred = predict(test_sims[sample])
    xyz = x[:, [names.index("x"), names.index("y"), names.index("z")]]
    shown = xyz[:, 1] <= 0.5 if half else np.ones(len(xyz), bool)
    lo, hi = true.min(), true.max()
    error = np.abs(pred - true)
    panels = [
        ("true temperature (°C)", true, "inferno", lo, hi),
        ("predicted temperature (°C)", pred, "inferno", lo, hi),
        ("absolute error (°C)", error, "viridis", 0, max(error.max(), 1e-9)),
    ]
    for i, (title, values, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        points = ax.scatter(
            *xyz[shown].T, c=values[shown], cmap=cmap, vmin=vmin, vmax=vmax, s=9
        )
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        fig.colorbar(points, ax=ax, shrink=0.6, pad=0.05)
    fig.text(
        0.5, 0.94,
        f"test simulation {sample}: {len(true)} nodes, mean error {error.mean():.3f} °C"
        + (" (cut at y=0.5)" if half else ""),
        ha="center", fontsize=12,
    )

    # --- all test meshes: predicted vs true ----------------------------------
    results = [predict(s) for s in test_sims]
    all_true = np.concatenate([r[1] for r in results])
    all_pred = np.concatenate([r[2] for r in results])
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(all_true, all_pred, s=2, alpha=0.3)
    lim = [all_true.min(), all_true.max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true (°C)", ylabel="predicted (°C)",
           title=f"{len(test_sims)} test simulations, MAE {np.abs(all_pred - all_true).mean():.3f} °C")

    ax = fig.add_subplot(2, 3, 5)
    ax.hist(all_pred - all_true, bins=60)
    ax.set(xlabel="error (°C)", ylabel="nodes", title="Error distribution")

    fig.subplots_adjust(top=0.9)
    return fig
