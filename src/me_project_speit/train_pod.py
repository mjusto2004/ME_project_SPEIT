"""Train and compare POD + regression surrogates.

Usage (from the repo root):
    python -m src.me_project_speit.train_pod --dataset src/me_project_speit/datasets/data_test.csv
"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.me_project_speit.models.pod import PODRegressor

# GP hyperparameters hitting their bounds is expected on noise-free FEM data
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Train POD + regression surrogates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n-modes", type=int, default=5)
    parser.add_argument("--regressors", nargs="+", choices=["gp", "ridge"], default=["gp", "ridge"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def split_indices(n, seed):
    """Same 70/15/15 split as train.py, so results are comparable with the MLP."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = n_val = max(1, round(n * 0.15))
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def metrics(Y_pred, Y_true):
    """MAE, RMSE and max absolute error, in °C."""
    err = Y_pred - Y_true
    return np.abs(err).mean(), np.sqrt((err**2).mean()), np.abs(err).max()


def main():
    args = parse_args()

    data = np.loadtxt(args.dataset, delimiter=",", ndmin=2)
    X, Y = data[:, :3], data[:, 3:]
    train_idx, val_idx, test_idx = split_indices(len(data), args.seed)
    # No early stopping here: train + val are both used for fitting
    fit_idx = np.concatenate([train_idx, val_idx])
    print(f"Samples: {len(data)} | fit: {len(fit_idx)} | test: {len(test_idx)}")

    run_dir = args.output_dir or (Path(__file__).resolve().parent / "runs"
                                  / f"pod_{args.dataset.stem}_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Baseline: always predict the mean training field
    rows.append(("mean baseline", *metrics(np.tile(Y[fit_idx].mean(0), (len(test_idx), 1)), Y[test_idx])))

    for name in args.regressors:
        model = PODRegressor(args.n_modes, name).fit(X[fit_idx], Y[fit_idx])

        if not rows[1:]:
            # Lower bound: error of the POD projection alone (perfect regression)
            pod = model.pod
            rows.append((f"POD projection (k={args.n_modes})",
                         *metrics(pod.decode(pod.encode(Y[test_idx])), Y[test_idx])))
            print("Cumulative explained variance:", np.round(np.cumsum(pod.explained_variance), 7))

        rows.append((f"POD + {name}", *metrics(model.predict(X[test_idx]), Y[test_idx])))
        with open(run_dir / f"pod_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Results table (test set, °C)
    print(f"\n{'model':<26}{'MAE':>10}{'RMSE':>10}{'MaxAE':>10}")
    for name, mae, rmse, maxae in rows:
        print(f"{name:<26}{mae:>10.4f}{rmse:>10.4f}{maxae:>10.4f}")
    np.savetxt(run_dir / "results.csv", np.array([r[1:] for r in rows]), delimiter=",",
               header="MAE,RMSE,MaxAE  rows: " + " | ".join(r[0] for r in rows))
    print(f"\nModels and results saved to: {run_dir}")


if __name__ == "__main__":
    main()
