"""Score several MLP checkpoints on the shared test sets and save the table."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from src.me_project_speit.models.mlp import MLP


def test_sets(baseline_ckpt, independent):
    """Return {name: (X, Y)}: the data_test.csv test split and the independent runs."""
    ckpt = torch.load(baseline_ckpt, map_location="cpu", weights_only=True)
    data = np.loadtxt(ckpt["dataset"], delimiter=",", ndmin=2)
    idx = ckpt["test_indices"].numpy()
    d = np.load(independent)
    return {
        "data_test.csv test split": (data[idx, :3], data[idx, 3:]),
        "independent random runs": (d["X"], d["T_final"]),
    }


def score(ckpt_path, X, Y):
    """MAE and RMSE (°C) of a checkpoint on (X, Y)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        x = (torch.as_tensor(X, dtype=torch.float32) - ckpt["x_mean"]) / ckpt["x_std"]
        pred = (model(x) * ckpt["y_std"] + ckpt["y_mean"]).numpy()
    err = pred - Y
    return np.abs(err).mean(), math.sqrt((err ** 2).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, metavar="NAME=CKPT")
    p.add_argument("--baseline-ckpt", type=Path, required=True,
                   help="checkpoint trained on data_test.csv; gives its test split")
    p.add_argument("--independent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    models = dict(m.split("=", 1) for m in args.model)
    rows = []
    for test_name, (X, Y) in test_sets(args.baseline_ckpt, args.independent).items():
        for model_name, path in models.items():
            mae, rmse = score(path, X, Y)
            rows.append([model_name, test_name, len(X), f"{mae:.4f}", f"{rmse:.4f}"])

    header = ["model", "tested on", "n test runs", "MAE (°C)", "RMSE (°C)"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        csv.writer(f).writerows([header] + rows)

    for r in [header] + rows:
        print(f"{r[0]:<36} {r[1]:<26} {str(r[2]):>11} {r[3]:>9} {r[4]:>10}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
