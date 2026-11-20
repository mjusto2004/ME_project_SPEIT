"""Report experiments on the fixed cube (field + pointwise datasets).

Learning curve: mean field, MLP, POD + ridge, POD + GP trained on 90 / 180 / 360 / 720
simulations, all scored on the same 100 independent simulations.

Usage: python report/scripts/exp_cube.py <train_720.csv> <independent_100.csv> [pointwise_360.csv]
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from me_project_speit.evaluation import evaluate
from me_project_speit.training import TrainConfig, train

OUT = Path("report/results")
RUNS = Path("runs/report")
OUT.mkdir(parents=True, exist_ok=True)


def subset(csv: Path, n: int) -> Path:
    """First n simulations of a field CSV, with its JSON."""
    dst = csv.with_name(f"cube_train_{n}.csv")
    if not dst.exists():
        lines = csv.read_text().splitlines(keepends=True)
        dst.write_text("".join(lines[:n]))
        spec = json.loads(csv.with_suffix(".json").read_text())
        spec["name"] = dst.stem
        dst.with_suffix(".json").write_text(json.dumps(spec, indent=2))
    return dst


def metrics(run_dir: Path, test: Path) -> dict:
    ev = evaluate(run_dir, test)
    ev.to_csv(OUT / f"{run_dir.name}_independent.csv")
    return ev.metrics()


def main():
    train_csv, test_csv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    Y_test = np.loadtxt(test_csv, delimiter=",")[:, 3:]
    for n in (90, 180, 360, 720):
        csv = subset(train_csv, n)
        Y = np.loadtxt(csv, delimiter=",")[:, 3:]
        err = np.abs(Y_test - Y.mean(axis=0))
        rows.append({"model": "mean", "n": n, "sim_MAE": float(err.mean(axis=1).mean()),
                     "RMSE": float(np.sqrt((err**2).mean())), "max_abs": float(err.max())})
        configs = {
            "mlp": dict(model="mlp", epochs=2000, patience=30),
            "mlp_cos": dict(model="mlp", epochs=1500, patience=1500, scheduler="cosine", warmup=10),
            "pod_ridge": dict(model="pod", model_kwargs={"regressor": "ridge"}),
            "pod_gp": dict(model="pod", model_kwargs={"regressor": "gp"}),
        }
        for name, kw in configs.items():
            run = RUNS / f"cube_{name}_{n}"
            if not (run / "best_model.pt").exists():
                train(TrainConfig(dataset=csv, output_dir=run, device="cuda" if "mlp" in name else "cpu", **kw))
            m = metrics(run, test_csv)
            rows.append({"model": name, "n": n, **m})
            print(name, n, m["sim_MAE"])
    if len(sys.argv) > 3:  # coordinate MLP on the pointwise dataset
        run = RUNS / "cube_coord_mlp_360"
        if not (run / "best_model.pt").exists():
            train(TrainConfig(dataset=Path(sys.argv[3]), model="coord_mlp", epochs=300, patience=30,
                              batch_size=4096, output_dir=run, device="cuda",
                              model_kwargs={"hidden": [256, 256, 256, 256]}))
        m = evaluate(run).metrics()  # test split of the pointwise data
        rows.append({"model": "coord_mlp_testsplit", "n": 360, **m})
    (OUT / "cube_learning_curve.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
