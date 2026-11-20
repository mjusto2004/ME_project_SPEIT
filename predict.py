"""Predict the transient temperature field of the aluminium cube and show it in 3D.

Usage (from the repository root)::

    python predict.py --T 293 --P 20000 --h 100                 # model 720 (default)
    python predict.py --T 293 --P 20000 --h 100 --model 100     # model trained on the 100 dataset
    python predict.py --T 298 --P 25000 --h 150 --model 360
    python predict.py --T 298 --P 25000 --h 150 --no-plot      # only print the statistics
    python predict.py --T 298 --P 25000 --h 150 --no-sim       # prediction only, no EasyFEA run

Models (--model): 100, 360, 720 (one MLP per dataset) or combined (trained on all three).

Inputs (physical units, same names as the training data):

    --T  initial temperature T0 (K)           range: 273.0 .. 303.0
    --P  heat flux Phi0 (W.m-2)               range: 15000.0 .. 30000.0
    --h  convection coefficient h (W.m-2.K-1) range: 50.0 .. 200.0

The model was trained on these ranges only. Values outside them are refused
unless ``--allow-extrapolation`` is given: the network then has to guess and
the result may be wrong.

The output is the temperature (deg C) at the 560 nodes of the cube mesh after
10 000 s. The 3D view needs EasyFEA and PyVista (``uv sync --extra sim``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # use the package from src/ without installing it

from me_project_speit.inference import Predictor  # noqa: E402

# --- Training ranges (inclusive). Keep in sync with simulations/config.py -----
RANGES = {
    "T": (273.0, 303.0),  # T0, initial temperature, K
    "P": (15000.0, 30000.0),  # Phi0, heat flux, W.m-2
    "h": (50.0, 200.0),  # convection coefficient, W.m-2.K-1
}

# --- Models: one MLP per dataset of data/new_ranges (choose with --model) ----
RUNS = ROOT / "src" / "me_project_speit" / "runs"
MODELS = {
    "100": RUNS / "best_model100.pt",  # trained on the 100-simulation dataset
    "360": RUNS / "best_model360.pt",  # trained on the 360-simulation dataset
    "720": RUNS / "best_model720.pt",  # trained on the 720-simulation dataset
    "combined": RUNS / "mlp_combined_100_360_720_20261120" / "best_model.pt",  # all 1180 simulations
}
DEFAULT_MODEL = "720"

# Fixed colour scale of the error view (°C), the same for every model so that the
# errors of the 100, 360 and 720 models can be compared. Errors above it are shown
# in the top colour. Change it with --error-max.
ERROR_MAX = 1.0


def check_ranges(values: dict[str, float], allow_extrapolation: bool) -> None:
    """Raise ``ValueError`` if a value is outside its training range (unless allowed)."""
    outside = [
        f"{name} = {values[name]:g} not in [{lo:g}, {hi:g}]"
        for name, (lo, hi) in RANGES.items()
        if not lo <= values[name] <= hi
    ]
    if outside and not allow_extrapolation:
        raise ValueError(
            "Input outside the training ranges: " + "; ".join(outside)
            + ". Use --allow-extrapolation to predict anyway (less reliable)."
        )
    if outside:
        print("WARNING: extrapolation, the prediction may be wrong: " + "; ".join(outside))


def predict_field(checkpoint: Path, T: float, P: float, h: float) -> np.ndarray:
    """Temperature (deg C) at the mesh nodes for the inputs T (K), P (W.m-2), h (W.m-2.K-1)."""
    predictor = Predictor(checkpoint)
    # Inputs must follow the order of the dataset: T0, Phi0, h.
    return predictor.predict(np.array([T, P, h], dtype=np.float32))[0]


def simulate_field(T: float, P: float, h: float) -> np.ndarray:
    """Temperature (deg C) at the mesh nodes from the EasyFEA simulation (about 0.5 s)."""
    from me_project_speit.simulations import CubeSimulation

    return CubeSimulation.run(T0=T, Phi0=P, h=h)


def show_comparison(
    predicted: np.ndarray,
    simulated: np.ndarray | None = None,
    title: str = "",
    error_max: float = ERROR_MAX,
) -> None:
    """Interactive 3D views on the cube mesh, side by side in one window.

    With ``simulated``: EasyFEA simulation | prediction | absolute error.
    Without it: the predicted field alone.
    """
    import pyvista as pv
    from EasyFEA import Models, PyVista, Simulations

    from me_project_speit.simulations import CubeSimulation, config

    mesh, _ = CubeSimulation.build_mesh()
    if len(predicted) != mesh.Nn or (simulated is not None and len(simulated) != mesh.Nn):
        raise ValueError(f"Expected {mesh.Nn} nodes per field.")
    simu = Simulations.Thermal(mesh, Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1))
    if simulated is None:
        views = [(predicted, "Prediction", None, "Temperature (°C)")]
    else:
        # Simulation and prediction share one colour scale so they can be compared directly.
        clim = (min(simulated.min(), predicted.min()), max(simulated.max(), predicted.max()))
        error = np.abs(predicted - simulated)
        views = [
            (simulated, "EasyFEA", clim, "Temperature (°C)"),
            (predicted, "Prediction", clim, "Temperature (°C)"),
            (error, "|Error|", (0.0, error_max), "|Error| (°C)"),
        ]

    # One window, one sub-plot per view; the cameras are linked so rotating one rotates all.
    plotter = pv.Plotter(shape=(1, len(views)), title=title, window_size=(560 * len(views), 620))
    for col, (values, label, lim, bar_title) in enumerate(views):
        plotter.subplot(0, col)
        PyVista.Plot(
            simu,
            values,
            plotMesh=True,
            nodeValues=True,
            clim=lim,
            colorbarTitle=bar_title,
            title=label,
            plotter=plotter,
        )
    plotter.link_views()
    plotter.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict and display the cube temperature field.")
    parser.add_argument("--T", type=float, required=True, help="initial temperature T0 (K), 273-303")
    parser.add_argument("--P", type=float, required=True, help="heat flux Phi0 (W.m-2), 15000-30000")
    parser.add_argument("--h", type=float, required=True, help="convection coefficient (W.m-2.K-1), 50-200")
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default=DEFAULT_MODEL,
        help="model trained on that dataset: 100, 360, 720 or combined (all three). Default: 720",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="use this best_model.pt instead of --model"
    )
    parser.add_argument("--no-plot", action="store_true", help="only print the statistics")
    parser.add_argument(
        "--error-max",
        type=float,
        default=ERROR_MAX,
        help=f"top of the fixed error colour scale in °C (default {ERROR_MAX}); same value = comparable plots",
    )
    parser.add_argument("--no-sim", action="store_true", help="skip the EasyFEA simulation (prediction only)")
    parser.add_argument(
        "--allow-extrapolation", action="store_true", help="accept inputs outside the training ranges"
    )
    args = parser.parse_args(argv)

    values = {"T": args.T, "P": args.P, "h": args.h}
    check_ranges(values, args.allow_extrapolation)

    checkpoint = args.checkpoint or MODELS[args.model]
    print(f"Model: {checkpoint}")
    predicted = predict_field(checkpoint, args.T, args.P, args.h)
    print(
        f"T0 = {args.T:g} K, Phi0 = {args.P:g} W.m-2, h = {args.h:g} W.m-2.K-1\n"
        f"Predicted temperature after 10 000 s: min {predicted.min():.2f} °C, "
        f"max {predicted.max():.2f} °C, mean {predicted.mean():.2f} °C"
    )

    simulated = None
    if not args.no_sim:
        simulated = simulate_field(args.T, args.P, args.h)
        err = predicted - simulated
        print(
            f"EasyFEA simulation:  min {simulated.min():.2f} °C, max {simulated.max():.2f} °C, "
            f"mean {simulated.mean():.2f} °C\n"
            f"Prediction error:    MAE {np.abs(err).mean():.3f} °C, RMSE {np.sqrt((err**2).mean()):.3f} °C, "
            f"max |error| {np.abs(err).max():.3f} °C, bias {err.mean():+.3f} °C"
        )

    if not args.no_plot:
        title = f"T0 = {args.T:g} K   Phi0 = {args.P:g} W/m²   h = {args.h:g} W/m²K"
        if simulated is None:
            show_comparison(predicted, title=title)  # prediction only: no simulation, no error view
        else:
            show_comparison(simulated, predicted, title, error_max=args.error_max)


if __name__ == "__main__":
    main()
