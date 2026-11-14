"""Score trained checkpoints on an independent test dataset and write a results table.

Usage: python evaluate_independent.py --test data/.../x_100.csv --out results.csv NAME=CKPT [NAME=CKPT ...]
Errors in deg C. Also reports error per T0 band and per h band.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from me_project_speit.datasets.base import load_dataset
from me_project_speit.inference import Predictor


def scores(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "max_abs": float(np.abs(err).max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("models", nargs="+", metavar="NAME=CKPT")
    args = p.parse_args()

    test = load_dataset(args.test)
    X, Y = np.asarray(test.X), np.asarray(test.Y)
    names = list(test.spec.input_names)
    T0, h = X[:, names.index("T0")], X[:, names.index("h")]
    bands = {
        "T0 < 293 K (cold start)": T0 < 293,
        "T0 >= 293 K (hot start)": T0 >= 293,
        "h < 100 (slow cooling)": h < 100,
        "h >= 100": h >= 100,
    }

    rows = []
    for item in args.models:
        name, ckpt = item.split("=", 1)
        pred = Predictor(ckpt)
        if list(pred.spec.input_names) != names:
            raise SystemExit(f"{name}: inputs {pred.spec.input_names} != test inputs {names}")
        Y_hat = pred.predict(X)
        rows.append({"model": name, "subset": f"all ({len(X)} runs)", **scores(Y, Y_hat)})
        for band, mask in bands.items():
            rows.append({"model": name, "subset": f"{band} ({int(mask.sum())} runs)",
                         **scores(Y[mask], Y_hat[mask])})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; not overwriting")
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Test set: {args.test} ({len(X)} runs), output spread "
          f"{Y.min():.2f}–{Y.max():.2f} °C, std {Y.std():.2f} °C")
    for r in rows:
        print(f"{r['model']:<10} {r['subset']:<34} MAE {r['MAE']:.4f}  "
              f"RMSE {r['RMSE']:.4f}  max {r['max_abs']:.4f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
