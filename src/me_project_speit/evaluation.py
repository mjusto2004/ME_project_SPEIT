"""Run a trained model on a dataset and measure its error, simulation by simulation.

Works for every dataset kind:

- ``field``: one row per simulation; node coordinates come from the reference
  cube mesh (needs EasyFEA) when the dataset lives on it;
- ``pointwise``: ``rows_per_sample`` rows per simulation, coordinates are inputs;
- ``graph``: one mesh per simulation, coordinates are inputs, edges from the file.

:func:`evaluate` returns an :class:`Evaluation`; :func:`plot_evaluation`
draws it (3D true / predicted / error, parity plot, error histogram, error per
simulation).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

COORD_NAMES = ("x", "y", "z")
NODE_INPUTS = {"x", "y", "z", "heated"}  # per-node inputs, not simulation parameters


@dataclass
class SimulationResult:
    """True and predicted values on the nodes of one simulation."""

    sim: int  # index of the simulation in the dataset
    params: dict[str, float]
    true: np.ndarray  # (n_nodes,)
    pred: np.ndarray  # (n_nodes,)
    coords: np.ndarray | None = None  # (n_nodes, 3) when known

    @property
    def error(self) -> np.ndarray:
        return self.pred - self.true

    @property
    def mae(self) -> float:
        return float(np.abs(self.error).mean())

    @property
    def rmse(self) -> float:
        return float(np.sqrt((self.error**2).mean()))

    @property
    def max_abs(self) -> float:
        return float(np.abs(self.error).max())

    @property
    def nmae(self) -> float:
        """MAE divided by the range (max - min) of the true field: 0.01 = 1 % of the range."""
        return self.mae / max(float(self.true.max() - self.true.min()), 1e-12)


@dataclass
class Evaluation:
    """Results of a model on (part of) a dataset."""

    checkpoint: Path
    model_name: str
    dataset: Path
    subset: str  # "all" or "test"
    results: list[SimulationResult] = field(default_factory=list)
    #: ``(min, max)`` of each parameter in the training data of the model.
    train_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def outside_training_ranges(self) -> dict[str, int]:
        """Number of simulations whose parameter lies outside the training range, per parameter."""
        counts = {}
        for name, (lo, hi) in self.train_ranges.items():
            n = sum(1 for r in self.results if name in r.params and not lo <= r.params[name] <= hi)
            if n:
                counts[name] = n
        return counts

    @property
    def true(self) -> np.ndarray:
        return np.concatenate([r.true for r in self.results])

    @property
    def pred(self) -> np.ndarray:
        return np.concatenate([r.pred for r in self.results])

    def metrics(self, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
        """Errors in physical units.

        Pooled over all node values: ``MAE``, ``RMSE``, ``max_abs``, ``bias``.

        Per simulation (each simulation weighs the same; the recommended
        figures to compare models, since the nodes of one simulation are not
        independent): ``sim_MAE`` (mean of the per-simulation MAEs) with its 95 %
        bootstrap confidence interval ``sim_MAE_lo``/``sim_MAE_hi`` (resampling
        the simulations), ``sim_MAE_median``, ``sim_MAE_p95``, and ``NMAE``
        (mean of MAE / range of the true field, as a fraction).
        """
        err = self.pred - self.true
        maes = np.array([r.mae for r in self.results])
        boot = np.random.default_rng(seed).choice(maes, size=(n_boot, len(maes))).mean(axis=1)
        return {
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err**2).mean())),
            "max_abs": float(np.abs(err).max()),
            "bias": float(err.mean()),
            "sim_MAE": float(maes.mean()),
            "sim_MAE_lo": float(np.percentile(boot, 2.5)),
            "sim_MAE_hi": float(np.percentile(boot, 97.5)),
            "sim_MAE_median": float(np.median(maes)),
            "sim_MAE_p95": float(np.percentile(maes, 95)),
            "NMAE": float(np.mean([r.nmae for r in self.results])),
        }

    def get(self, sim: int) -> SimulationResult:
        for r in self.results:
            if r.sim == sim:
                return r
        raise ValueError(
            f"Simulation {sim} is not in the evaluated set "
            f"({len(self.results)} simulations: {_short([r.sim for r in self.results])})."
        )

    def worst(self) -> SimulationResult:
        return max(self.results, key=lambda r: r.mae)

    def summary(self, top: int = 5) -> str:
        """Text report: overall metrics, then the worst and best simulations."""
        m = self.metrics()
        true = self.true
        lines = [
            f"Model:       {self.model_name} ({self.checkpoint})",
            f"Dataset:     {self.dataset} ({self.subset}: {len(self.results)} simulations, "
            f"{len(true)} node values)",
            f"Targets:     {true.min():.3f} .. {true.max():.3f}, std {true.std():.3f}",
            "Per simulation (each simulation weighs the same):",
            f"  MAE:         {m['sim_MAE']:.4f}  (95 % CI {m['sim_MAE_lo']:.4f} .. {m['sim_MAE_hi']:.4f})",
            f"  median:      {m['sim_MAE_median']:.4f}   95th percentile: {m['sim_MAE_p95']:.4f}",
            f"  relative:    {100 * m['NMAE']:.2f} % of the field's range (NMAE)",
            "Over all node values:",
            f"  MAE:         {m['MAE']:.4f}",
            f"  RMSE:        {m['RMSE']:.4f}",
            f"  Max |error|: {m['max_abs']:.4f}",
            f"  Bias:        {m['bias']:+.4f} (mean of predicted - true)",
        ]
        for name, n in self.outside_training_ranges().items():
            lo, hi = self.train_ranges[name]
            lines.append(
                f"Warning:     {n}/{len(self.results)} simulations have {name} outside the "
                f"training range [{lo:.4g}, {hi:.4g}]: the model extrapolates there."
            )
        ranked = sorted(self.results, key=lambda r: r.mae, reverse=True)
        top = min(top, len(ranked))
        for title, items in (("Worst", ranked[:top]), ("Best", ranked[::-1][:top])):
            lines.append(f"{title} simulations (by MAE):")
            for r in items:
                params = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
                lines.append(
                    f"  sim {r.sim:>4}  MAE {r.mae:.4f}  NMAE {100 * r.nmae:5.2f} %  "
                    f"max {r.max_abs:.4f}  ({params})"
                )
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> Path:
        """One row per simulation: id, parameters, MAE, NMAE, RMSE, max |error|."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.results[0].params)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim", *names, "n_nodes", "MAE", "NMAE", "RMSE", "max_abs"])
            for r in self.results:
                w.writerow([r.sim, *(r.params[n] for n in names), len(r.true),
                            r.mae, r.nmae, r.rmse, r.max_abs])
        return path


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def evaluate(
    checkpoint: str | Path,
    dataset: str | Path | None = None,
    subset: str = "auto",
) -> Evaluation:
    """Predict every simulation of ``dataset`` with ``checkpoint`` and keep the errors.

    Parameters
    ----------
    checkpoint : ``best_model.pt`` or the run folder containing it.
    dataset : CSV of the dataset to infer on. ``None``: the training dataset.
    subset : ``"all"`` (every simulation), ``"test"`` (the test split of the
        run; only for its own training dataset) or ``"auto"``: ``"test"`` on
        the training dataset, ``"all"`` on any other.
    """
    from .datasets import load_dataset
    from .inference import Predictor
    from .training import _check_same_data

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at '{checkpoint}'.")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

    train_path = Path(ckpt["dataset"])
    dataset = train_path if dataset is None else Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset '{dataset}' not found.")
    is_training_data = dataset.resolve() == train_path.resolve()
    if subset == "auto":
        subset = "test" if is_training_data else "all"
    if subset not in ("all", "test"):
        raise ValueError("subset must be 'all', 'test' or 'auto'")
    if subset == "test" and not is_training_data:
        raise ValueError(
            f"'test' subset only exists for the training dataset of the run ({train_path})."
        )

    data = load_dataset(dataset)
    predictor = Predictor(checkpoint)
    _check_same_data(predictor.spec, data.spec, dataset.name)

    sims = range(len(data))
    if subset == "test":
        sims = _test_simulations(data, ckpt["test_indices"].numpy())

    results = [_predict_simulation(data, predictor, s) for s in sims]
    ignored = set(getattr(predictor.model, "ignored_inputs", ()))
    ranges = {k: tuple(map(float, v)) for k, v in predictor.spec.input_ranges.items() if k not in ignored}
    return Evaluation(checkpoint, ckpt["model_name"], dataset, subset, results, ranges)


def _test_simulations(data, test_rows: np.ndarray) -> list[int]:
    if data.spec.kind == "graph":
        return sorted(set(data.sim[test_rows].tolist()))
    return sorted(set((test_rows // data.spec.rows_per_sample).tolist()))


def _predict_simulation(data, predictor, s: int) -> SimulationResult:
    names = data.spec.input_names
    kind = data.spec.kind

    if kind == "graph":
        a, b = data.node_start[s], data.node_stop[s]
        c, d = data.edge_start[s], data.edge_stop[s]
        edges = np.stack([data.edge_src[c:d], data.edge_dst[c:d]])
        x, true = data.X[a:b], data.Y[a:b, 0]
        pred = predictor.predict(x, edges)[:, 0]
    elif data.spec.rows_per_sample == 1:  # one row = one whole field
        x, true = data.X[s : s + 1], data.Y[s]
        pred = predictor.predict(x)[0]
    else:  # pointwise-like: rows_per_sample rows of one output each
        n = data.spec.rows_per_sample
        x, true = data.X[s * n : (s + 1) * n], data.Y[s * n : (s + 1) * n, 0]
        pred = predictor.predict(x)[:, 0]

    params = {k: float(x[0, i]) for i, k in enumerate(names) if k not in NODE_INPUTS}
    if all(c in names for c in COORD_NAMES):
        coords = x[:, [names.index(c) for c in COORD_NAMES]]
    else:
        coords = _mesh_coordinates(data.spec, len(true))
    return SimulationResult(s, params, true.astype(float), pred.astype(float), coords)


def _mesh_coordinates(spec, n_nodes: int) -> np.ndarray | None:
    """Node coordinates of a ``field`` dataset living on the reference cube mesh, else ``None``."""
    from .simulations import config

    if spec.mesh_id != config.MESH_ID or n_nodes != config.N_NODES:
        return None
    try:
        from .datasets.pointwise import _node_coordinates

        return _node_coordinates()
    except ImportError:  # EasyFEA missing: metrics only
        return None


def _short(ids: list[int], n: int = 10) -> str:
    return ", ".join(map(str, ids[:n])) + (", ..." if len(ids) > n else "")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_evaluation(
    ev: Evaluation,
    sim: int | None = None,
    half: bool = False,
    save: str | Path | None = None,
) -> list[Path]:
    """One figure: 3D true / predicted / error of simulation ``sim`` (default: the worst),
    then predicted vs true, the error histogram and the MAE of every simulation.

    ``half`` keeps only the nodes with y <= 0.5 to see inside. With ``save``, the
    figure is written to ``<save>_inference.png`` instead of being shown.
    Returns the saved files.
    """
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = draw_evaluation(plt, ev, ev.worst() if sim is None else ev.get(sim), half)
    if save is None:
        plt.show()
        return []
    path = Path(f"{save}_inference.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return [path]


def draw_evaluation(plt, ev: Evaluation, chosen: SimulationResult, half: bool = False, label: str | None = None):
    """Build the figure of :func:`plot_evaluation` (``label`` names ``chosen`` in the title)."""
    fig = plt.figure(figsize=(15, 8))
    label = label or f"simulation {chosen.sim}"

    # --- one simulation: true, predicted, error ------------------------------
    if chosen.coords is not None:
        xyz = chosen.coords
        shown = xyz[:, 1] <= 0.5 if half else np.ones(len(xyz), bool)
        lo = min(chosen.true.min(), chosen.pred.min())  # shared scale: a biased prediction shows
        hi = max(chosen.true.max(), chosen.pred.max())
        error = np.abs(chosen.error)
        panels = [
            ("true temperature (°C)", chosen.true, "inferno", lo, hi),
            ("predicted temperature (°C)", chosen.pred, "inferno", lo, hi),
            ("absolute error (°C)", error, "viridis", 0, max(error.max(), 1e-9)),
        ]
        for i, (title, values, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            points = ax.scatter(*xyz[shown].T, c=values[shown], cmap=cmap, vmin=vmin, vmax=vmax, s=9)
            ax.set_title(title)
            ax.set_box_aspect((1, 1, 1))
            fig.colorbar(points, ax=ax, shrink=0.6, pad=0.05)
    else:
        ax = fig.add_subplot(2, 3, (1, 3))
        ax.plot(chosen.true, label="true")
        ax.plot(chosen.pred, label="predicted", alpha=0.8)
        ax.set(xlabel="output index (no node coordinates for this mesh)", ylabel="temperature (°C)")
        ax.legend()
    params = ", ".join(f"{k}={v:.4g}" for k, v in chosen.params.items())
    fig.text(
        0.5, 0.95,
        f"{label} ({params}): {len(chosen.true)} nodes, MAE {chosen.mae:.3f} °C, "
        f"max {chosen.max_abs:.3f} °C" + (" (cut at y=0.5)" if half else ""),
        ha="center", fontsize=11,
    )

    # --- all simulations ------------------------------------------------------
    m = ev.metrics()
    all_true, all_pred = ev.true, ev.pred
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(all_true, all_pred, s=2, alpha=0.3)
    lim = [min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true (°C)", ylabel="predicted (°C)",
           title=f"{len(ev.results)} simulations, MAE {m['sim_MAE']:.3f} °C "
                 f"[{m['sim_MAE_lo']:.3f}, {m['sim_MAE_hi']:.3f}]")

    ax = fig.add_subplot(2, 3, 5)
    ax.hist(all_pred - all_true, bins=60)
    ax.set(xlabel="predicted - true (°C)", ylabel="nodes", title=f"Error distribution (RMSE {m['RMSE']:.3f} °C)")

    ax = fig.add_subplot(2, 3, 6)
    maes = np.array([r.mae for r in ev.results])
    colors = ["C3" if r is chosen else "C0" for r in ev.results]
    ax.bar(np.arange(len(maes)), maes, color=colors, width=1.0)
    ax.set(xlabel="simulation (evaluation order)", ylabel="MAE (°C)",
           title=f"Error per simulation (red: {label})")

    fig.suptitle(f"{ev.checkpoint.parent.name} ({ev.model_name}) on {ev.dataset.stem} [{ev.subset}]", y=0.995)
    fig.subplots_adjust(top=0.9, hspace=0.35)
    return fig


def show_on_mesh(result: SimulationResult) -> None:
    """Interactive PyVista views (true, predicted, error) of a simulation on the reference
    cube mesh (``field`` and ``pointwise`` datasets; needs EasyFEA)."""
    from EasyFEA import Models, PyVista, Simulations

    from .simulations import CubeSimulation, config

    mesh, _ = CubeSimulation.build_mesh()
    if len(result.true) != mesh.Nn:
        raise ValueError(
            f"Simulation {result.sim} has {len(result.true)} nodes, the reference mesh {mesh.Nn}: "
            "the interactive mesh view only exists for the cube datasets (field, pointwise)."
        )
    simu = Simulations.Thermal(mesh, Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1))
    clim = (result.true.min(), result.true.max())
    for values, title, lim in (
        (result.true, "True temperature (°C)", clim),
        (result.pred, "Predicted temperature (°C)", clim),
        (np.abs(result.error), "Absolute error (°C)", None),
    ):
        PyVista.Plot(simu, values, plotMesh=True, nodeValues=True, clim=lim, colorbarTitle=title).show()
