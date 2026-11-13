# Experiments

Experiments 1–2 use the baseline MLP (`src/me_project_speit/models/mlp.py`). Same MLP and training loop
throughout: Adam on MSE, early stopping with patience 30, max 2000 epochs, unless stated otherwise.
Experiment 3 compares it with a POD + regression surrogate (`src/me_project_speit/models/pod.py`).

**Result:** trained on 360 runs instead of 90, the MLP reaches MAE 0.0076 °C instead of 0.039 °C
(100 independent test runs).

## Test sets

| Test set | Runs | Source |
|---|---|---|
| `data_test.csv` test split | 14 | 15 % test split of `data_test.csv` (seed 42) used by `train.py` |
| Independent runs | 100 | `datasets/independent_test_100.npz`, new random runs (seed 2) |

Inputs are sampled uniformly as in `data_generation.py`: T0 ∈ [146.5, 293] K,
P ∈ [15 000, 30 000] W, h ∈ [150, 300] W·m⁻²·K⁻¹. Errors in °C.

## Experiment 1 — More random simulations (adopted)

**Setup:** new random runs (seed 1) are added to the 90 runs of `data_test.csv`.
The original 14 test runs stay held out; the new runs go 85 % to training, 15 % to validation.

| Training data | Runs | `data_test.csv` split MAE / RMSE | Independent MAE / RMSE |
|---|---|---|---|
| `data_test.csv` (baseline) | 90 | 0.0304 / 0.0491 | 0.0387 / 0.0685 |
| + 90 random | 180 | 0.0251 / 0.0380 | 0.0251 / 0.0487 |
| **+ 270 random** | **360** | **0.0079 / 0.0104** | **0.0076 / 0.0115** |
| + 630 random | 720 | 0.0071 / 0.0098 | 0.0075 / 0.0116 |

**Conclusion:** error drops up to 360 runs; 720 runs give no further gain.
`datasets/data_test_extended_360.npz` is the adopted training set.

## Experiment 2 — Sweep of h with 2 runs per h (rejected)

**Idea:** at fixed h the solver is linear in T0 and P: T = T_ext + a(h)·(T0 − T_ext) + b(h)·P.
Two runs per h fix a(h) and b(h), so only h needs dense sampling.

**Setup:** 40 evenly spaced h values, 3 runs each (2 fixed (T0, P) corners + 1 random check),
120 runs. Train / val / test split by h value (24 / 8 / 8 h values).

| Model | `data_test.csv` split MAE | Independent MAE |
|---|---|---|
| MLP trained on h-sweep (120 runs), patience 30 | 0.0928 | 0.0788 |
| MLP trained on h-sweep (120 runs), no early stopping | 0.0520 | 0.0581 |
| MLP trained on `data_test.csv` (90 runs) | 0.0304 | 0.0387 |

**Conclusion:** linearity confirmed (check run predicted within 1e-12 °C), but the MLP does worse:
the data covers (T0, P) too thinly. Rejected; code removed.

## Experiment 3 — POD + regression surrogate

**Model** (`models/pod.py`): T ≈ T̄ + Σᵢ cᵢ φᵢ with k = 5 POD modes (SVD of the training fields),
then a regressor (T0, P, h) → (c₁, …, c₅).
- `gp`: one Gaussian process per mode, anisotropic RBF kernel.
- `ridge`: RidgeCV on features T0, P, h, P/h, T0/h, 1/h, P·h, T0·h, h².

**Setup:** same splits as the MLP; train + val both used for fitting (no early stopping).
3 modes explain 99.9998 % of the variance; projection onto 5 modes alone gives MAE < 1e-4 °C.

| Model | Runs | `data_test.csv` split MAE / RMSE | Independent MAE / RMSE |
|---|---|---|---|
| Mean field | 90 | 0.2806 / 0.3593 | — |
| MLP | 90 | 0.0304 / 0.0491 | 0.0387 / 0.0685 |
| POD + ridge | 90 | 0.0078 / 0.0102 | — |
| POD + GP | 90 | 0.0001 / 0.0001 | — |
| MLP | 360 | 0.0079 / 0.0104 | 0.0076 / 0.0115 |
| POD + ridge | 360 | 0.0084 / 0.0098 | 0.0092 / 0.0123 |
| **POD + GP** | **360** | **0.000014 / 0.000018** | **0.000013 / 0.000020** |

**Conclusion:** POD + GP is ~500× more accurate than the MLP at 360 runs and reaches the POD
projection limit (max error 1e-4 °C); it trains in seconds on CPU. Ridge is capped by its features
and does not improve with more data.

Run: `python train_pod.py --dataset src/me_project_speit/datasets/data_test.csv` (CSV only; the
360-run results come from the same `PODRegressor` fit on the non-test rows of the `.npz`).
Outputs in `runs/pod_<dataset>_<time>/`.

## Reproduce

Code is in `src/me_project_speit/extended_dataset/`.

| File | Role |
|---|---|
| `simulate.py` | Same physics as `simulation.py`, also returns the time history |
| `generate.py` | Random runs in parallel, one `.npz` per run in `datasets/raw/extended_runs/` |
| `build.py` | `data_test.csv` + N new runs → `data_test_extended_<total>.npz`, keeps the original test split |
| `train.py` | Same MLP and loop as the baseline `train.py`, uses the split saved in the `.npz` |
| `compare.py` | Scores several checkpoints on both test sets |

Full pipeline on a Slurm cluster (set the paths in the script first):

```
sbatch scripts/slurm/extended_dataset.sbatch
```

| Output | Content |
|---|---|
| `datasets/data_test_extended_360.npz` | Training set (X, T_final, split) |
| `datasets/independent_test_100.npz` | Independent test set (X, T_final) |
| `datasets/raw/extended_runs/` | One file per run: inputs, final field, time history, mesh (git-ignored) |
| `runs/mlp_trained_on_<dataset>_<time>/` | `best_model.pt`, `history.csv` |
| `runs/comparison_<time>.csv` | Results table |

## Open issue — node numbering of `coord.npy`

| | |
|---|---|
| **Issue** | 85 of 560 nodes in `coord.npy` are ordered differently from the columns of `data_test.csv`. |
| **Presumed cause** | Node numbering depends on machine/version: `data_test.csv` = Linux (EasyFEA 7.2.0), `coord.npy` = macOS (EasyFEA 7.1.0). |
| **Affected** | `datasets/observe.py` (wrong node positions in plots); future models using `coord.npy` (e.g. CNN on an interpolated grid). |
| **Not affected** | Training results above (all generated on Linux, same numbering as `data_test.csv`). |
| **In place** | `extended_dataset/build.py` checks node order before combining. |
| **Proposed fix** | Replace `coord.npy` with coordinates matching `data_test.csv`; align nodes by coordinates in `observe.py`. |
