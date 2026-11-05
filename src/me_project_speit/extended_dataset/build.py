"""Build data_test.csv + N extra runs, keeping the original 14 test runs as the test set.

New runs are reordered to the node order of data_test.csv, which differs from the
current mesh for some nodes (see docs/experiments.md).
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import torch

from src.me_project_speit.extended_dataset.generate import DEFAULT_OUT
from src.me_project_speit.extended_dataset.simulate import run


def node_order(base, rows=(0, 45)):
    """Index array `order` such that a field from the current mesh, taken as field[order],
    is in data_test.csv column order. Found by re-simulating some rows of data_test.csv."""
    order = None
    for i in rows:
        with contextlib.redirect_stdout(io.StringIO()):
            _, T_hist, coord = run(*base[i, :3])
        sim, ref = T_hist[-1], base[i, 3:]
        if order is None:
            order = np.empty(len(ref), dtype=int)
            order[np.argsort(ref)] = np.argsort(sim)
        err = np.abs(sim[order] - ref).max()
        if err > 1e-8:
            raise RuntimeError(f"Row {i} of data_test.csv not reproduced (max error {err:.2e} °C)")
    return order, coord[order]


def load_runs(files, order):
    X = np.array([[float(np.load(f)[k]) for k in ("T0", "P", "h")] for f in files])
    Y = np.array([np.load(f)["T_final"][order] for f in files])
    return X, Y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-ckpt", type=Path, required=True,
                   help="checkpoint trained on data_test.csv; gives the original split")
    p.add_argument("--n-extra", type=int, nargs="+", required=True)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT.parent.parent)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=True)
    base = np.loadtxt(ckpt["dataset"], delimiter=",", ndmin=2)
    base_split = np.full(len(base), "train", dtype="<U5")
    base_split[ckpt["val_indices"].numpy()] = "val"
    base_split[ckpt["test_indices"].numpy()] = "test"

    out = args.out_dir
    order, coord = node_order(base)
    np.save(out / "node_order_data_test.npy", order)
    print(f"Node order: {int(np.sum(order != np.arange(len(order))))} of {len(order)} nodes "
          f"reordered to match data_test.csv")

    X_ind, Y_ind = load_runs(sorted(args.in_dir.glob("test_*.npz")), order)
    ind_path = out / f"independent_test_{len(X_ind)}.npz"
    np.savez_compressed(ind_path, X=X_ind, T_final=Y_ind, coord=coord)
    print(f"{ind_path.name}: {len(X_ind)} runs")

    X_new, Y_new = load_runs(sorted(args.in_dir.glob("train_*.npz")), order)
    for n in args.n_extra:
        if n > len(X_new):
            raise SystemExit(f"Only {len(X_new)} extra runs available, asked for {n}")
        rng = np.random.default_rng([args.seed, n])
        new_split = np.where(rng.random(n) < 0.15, "val", "train")
        X = np.vstack([base[:, :3], X_new[:n]])
        Y = np.vstack([base[:, 3:], Y_new[:n]])
        split = np.concatenate([base_split, new_split])
        path = out / f"data_test_extended_{len(X)}.npz"
        np.savez_compressed(path, X=X, T_final=Y, split=split, coord=coord)
        counts = {s: int(np.sum(split == s)) for s in ("train", "val", "test")}
        print(f"{path.name}: {len(X)} runs {counts}")


if __name__ == "__main__":
    main()
