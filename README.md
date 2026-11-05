# ME_project_SPEIT

**Neural surrogate models for transient 3D thermal finite-element simulations**

Generates transient thermal simulations with [EasyFEA](https://github.com/matnoel/EasyFEA), trains a neural network mapping the inputs `(T0, P, h)` to the final temperature field on the mesh (560 nodes), and visualizes the results.

## Contents

```
ME_project_SPEIT/
├── src/me_project_speit/
│   ├── simulation.py          # EasyFEA thermal simulation
│   ├── data_generation.py     # Runs simulations and writes a CSV dataset
│   ├── train.py               # Training entry point
│   ├── models/
│   │   └── mlp.py             # Baseline MLP: 3 inputs -> 560 temperatures
│   ├── extended_dataset/      # More random runs + retraining (see docs/experiments.md)
│   ├── datasets/
│   │   ├── data_test.csv      # Example dataset (90 simulations)
│   │   ├── data_test_extended_360.npz  # 90 + 270 random runs (recommended)
│   │   ├── independent_test_100.npz    # Shared test set (100 runs)
│   │   ├── coord.npy          # Mesh node coordinates
│   │   └── observe.py         # Plots stored simulations on the mesh
│   └── runs/                  # Training outputs (auto-created)
├── docs/experiments.md        # Experiment log: setup, equations, results
├── scripts/slurm/             # Slurm job for the extended dataset
└── resources/                 # Reference scripts, notebooks, Slurm script
```

## Requirements

Python 3.12 or later. Run all commands from the repository root.

```
pip install EasyFEA torch numpy matplotlib
```

## Data Generation

```
python -m src.me_project_speit.data_generation
```

The number of simulations is set by `save_dataset(N)` in the `__main__` block. The dataset is saved to `src/me_project_speit/datasets/` as a headerless CSV with one row per simulation: columns 0–2 are the inputs `T0`, `P`, `h`, and columns 3–562 are the final nodal temperatures (°C).

## Training

```
python -m src.me_project_speit.train --dataset src/me_project_speit/datasets/data_test.csv
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to the CSV dataset |
| `--epochs` | 300 | Maximum number of epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 30 | Early-stopping patience |
| `--seed` | 42 | Random seed |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--output-dir` | `src/me_project_speit/runs/<dataset>_<timestamp>` | Output directory |

Each run saves `best_model.pt` (weights + normalization stats) and `history.csv` (per-epoch metrics), and prints the test MAE / RMSE.

## Adding a New Model

A model is any `torch.nn.Module` that takes a `(batch, 3)` tensor and returns a `(batch, 560)` tensor. `train.py` handles normalization, loss, early stopping and checkpointing.

1. Create the model class in `src/me_project_speit/models/`, e.g. `my_model.py` with class `MyModel`.
2. In `src/me_project_speit/train.py`, replace the MLP import and instantiation:

   ```python
   from src.me_project_speit.models.my_model import MyModel
   ...
   model = MyModel().to(device)
   ```

3. Train it with the same command as above.

## Extended Dataset

The same MLP trained on 360 runs (`data_test.csv` + 270 random runs) reaches a test MAE of 0.0076 °C, against 0.039 °C with 90 runs. Full pipeline (generate, build, train, compare) on a Slurm cluster, after updating its paths:

```
sbatch scripts/slurm/extended_dataset.sbatch
```

Setup, equations and results: [`docs/experiments.md`](docs/experiments.md).

## Visualization

```
python src/me_project_speit/datasets/observe.py
```

Edit `ids` in the script to choose which simulations to plot.

## Running on Slurm

`resources/jupyter_gpu.sbatch` launches Jupyter Lab on a GPU node. Update its paths, then run `sbatch resources/jupyter_gpu.sbatch`.
