"""Limits of the cube surrogates: extrapolation in h, very small training sets, speed."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from me_project_speit.evaluation import evaluate
from me_project_speit.inference import Predictor
from me_project_speit.training import TrainConfig, train

D = Path("data/report")
OUT = Path("report/results")
RUNS = Path("runs/report")
train_all = D / "cube_train_all720.csv"
test = D / "cube_independent_100.csv"


def write_subset(rows, name):
    lines = [l for l in rows]
    dst = D / f"{name}.csv"
    dst.write_text("".join(lines))
    spec = json.loads(train_all.with_suffix(".json").read_text())
    spec["name"] = name
    dst.with_suffix(".json").write_text(json.dumps(spec, indent=2))
    return dst


def per_sim(run, csv):
    ev = evaluate(run, csv)
    return {r.sim: r.mae for r in ev.results}


lines = train_all.read_text().splitlines(keepends=True)
h = np.array([float(l.split(",")[2]) for l in lines])
res = {}

# 1. extrapolation: train only on h >= 100, look at test simulations with h < 100
hi = write_subset([l for l, v in zip(lines, h) if v >= 100], "cube_train_hge100")
h_test = np.loadtxt(test, delimiter=",")[:, 2]
for name, kw in {
    "mlp": dict(model="mlp", epochs=1500, patience=1500, scheduler="cosine", warmup=10, device="cuda"),
    "pod_ridge": dict(model="pod", model_kwargs={"regressor": "ridge"}, device="cpu"),
    "pod_gp": dict(model="pod", model_kwargs={"regressor": "gp"}, device="cpu"),
}.items():
    run = RUNS / f"cube_extrap_{name}"
    if not (run / "best_model.pt").exists():
        train(TrainConfig(dataset=hi, output_dir=run, **kw))
    maes = per_sim(run, test)
    inside = [m for s, m in maes.items() if h_test[s] >= 100]
    outside = [m for s, m in maes.items() if h_test[s] < 100]
    far = [m for s, m in maes.items() if h_test[s] < 70]
    res[f"extrap_{name}"] = {"inside": float(np.mean(inside)), "outside": float(np.mean(outside)),
                             "far_h_lt_70": float(np.mean(far)), "n_in": len(inside), "n_out": len(outside),
                             "n_far": len(far)}
    np.savetxt(OUT / f"extrap_{name}.txt", np.array([[h_test[s], m] for s, m in maes.items()]))
    print(name, res[f"extrap_{name}"])

# 2. how few simulations does POD + GP need?
for n in (10, 20, 45):
    csv = write_subset(lines[:n], f"cube_train_{n}")
    for name in ("gp", "ridge"):
        run = RUNS / f"cube_pod_{name}_{n}"
        if not (run / "best_model.pt").exists():
            # tiny sets: the 70/15/15 split leaves ~70 % of them for fitting
            train(TrainConfig(dataset=csv, model="pod", model_kwargs={"regressor": name,
                                                                        "n_modes": min(5, int(0.7 * n) - 1)},
                              output_dir=run, device="cpu"))
        m = evaluate(run, test).metrics()
        res[f"pod_{name}_{n}"] = m["sim_MAE"]
        print(name, n, m["sim_MAE"])

# 3. speed: one prediction (batch of 1) and 100 predictions
X = np.loadtxt(test, delimiter=",")[:, :3].astype(np.float32)
for name in ("mlp_cos_720", "pod_gp_720", "pod_ridge_720"):
    p = Predictor(RUNS / f"cube_{name}" / "best_model.pt")
    p.predict(X[:1])
    t = time.perf_counter()
    for i in range(100):
        p.predict(X[i : i + 1])
    one = (time.perf_counter() - t) / 100
    t = time.perf_counter()
    p.predict(X)
    batch = time.perf_counter() - t
    res[f"time_{name}"] = {"one_ms": one * 1e3, "batch100_ms": batch * 1e3}
    print(name, res[f"time_{name}"])

(OUT / "cube_limits.json").write_text(json.dumps(res, indent=2))
