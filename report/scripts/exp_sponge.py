"""Sponge experiments: GNN vs GNO trained on coarse meshes (res 5-10), tested at res 7, 12, 18, 25."""

import json
import sys
from pathlib import Path

import numpy as np

from me_project_speit.evaluation import evaluate
from me_project_speit.training import TrainConfig, train

D = Path("data/report")
OUT = Path("report/results")
RUNS = Path("runs/report")
TRAIN = D / "sponge_train_1000.csv"
TESTS = {res: D / f"sponge_test_res{res}.csv" for res in (7, 12, 18, 25)}

CONFIGS = {
    "gnn": dict(model="gnn", model_kwargs={"hidden": 64, "layers": 4}, batch_size=4),
    "gnn8": dict(model="gnn", model_kwargs={"hidden": 64, "layers": 8}, batch_size=4),
    "gno": dict(model="gno", batch_size=4),
}
names = sys.argv[1:] or list(CONFIGS)
results = json.loads((OUT / "sponge_results.json").read_text()) if (OUT / "sponge_results.json").exists() else {}

# baseline: predict the mean training temperature everywhere
data = np.loadtxt(TRAIN, delimiter=",", usecols=(0, 10))
mean_T = float(data[:, 1].mean())
for res, csv in [("split", None), *TESTS.items()]:
    if csv is None or not csv.exists():
        continue
    d = np.loadtxt(csv, delimiter=",", usecols=(0, 10))
    maes = [np.abs(d[d[:, 0] == s, 1] - mean_T).mean() for s in np.unique(d[:, 0])]
    results.setdefault("mean", {})[str(res)] = {"sim_MAE": float(np.mean(maes))}

for name in names:
    run = RUNS / f"sponge_{name}"
    if not (run / "best_model.pt").exists():
        train(TrainConfig(dataset=TRAIN, output_dir=run, device="cuda", epochs=100, patience=100,
                          scheduler="cosine", warmup=3, **CONFIGS[name]))
    ev = evaluate(run)  # test split (150 simulations, res 5-10)
    ev.to_csv(OUT / f"sponge_{name}_split.csv")
    results.setdefault(name, {})["split"] = ev.metrics()
    for res, csv in TESTS.items():
        if csv.exists():
            ev = evaluate(run, csv)
            ev.to_csv(OUT / f"sponge_{name}_res{res}.csv")
            results[name][str(res)] = ev.metrics()
    print(name, {k: round(v["sim_MAE"], 3) for k, v in results[name].items()})
    (OUT / "sponge_results.json").write_text(json.dumps(results, indent=2))
(OUT / "sponge_results.json").write_text(json.dumps(results, indent=2))
