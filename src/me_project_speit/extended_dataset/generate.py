"""Run random simulations with the same input ranges as data_generation.py, one .npz per run."""

import argparse
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

T0_RANGE = (146.5, 293.0)
P_RANGE = (15000.0, 30000.0)
H_RANGE = (150.0, 300.0)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "extended_runs"


def run_one(job):
    """Run one simulation and save it; skip if the file already exists."""
    from src.me_project_speit.extended_dataset.simulate import run

    name, seed, i, out_dir = job
    path = Path(out_dir) / f"{name}_{i:04d}.npz"
    if path.exists():
        return f"skip {path.name}"

    rng = np.random.default_rng([seed, i])
    T0, P, h = rng.uniform(*T0_RANGE), rng.uniform(*P_RANGE), rng.uniform(*H_RANGE)
    with contextlib.redirect_stdout(io.StringIO()):
        times, T_hist, coord = run(T0, P, h)

    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, T0=T0, P=P, h=h, seed=seed, times=times,
                        T_final=T_hist[-1], T_hist=T_hist.astype(np.float32), coord=coord)
    tmp.rename(path)
    return f"done {path.name}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="file prefix, e.g. train or test")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.name, args.seed, i, str(args.out_dir)) for i in range(args.n)]
    print(f"{len(jobs)} runs, {args.workers} workers -> {args.out_dir}")
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        for k, msg in enumerate(pool.map(run_one, jobs), 1):
            if k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
